/* Front-end Integra-LOST.
   Сборка не нужна: чистый ES2020 в браузере. */

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const state = {
  loaded: false,
  playing: false,
  activeStreamId: "main",
  currentPage: "monitor",
  streams: [],
  totalFrames: 0,
  fps: 30,
  seekDragging: false,
  // Храним активные тревоги в памяти UI: ключ = track_id.
  alarms: new Map(),
  ws: null,
  settings: null,
  metricsTimer: null,
  streamsTimer: null,
  theme: "light",
};

const EVENT_UI = {
  abandoned: { title: "Обнаружен предмет", desc: "Подозрительный предмет оставлен без владельца." },
  disappeared: { title: "Предмет пропал", desc: "Ранее обнаруженный предмет исчез из кадра." },
};

// ----------------------------------------------------------- вспомогательные функции

function fmtTime(sec) {
  if (!isFinite(sec)) return "00:00.00";
  const m = Math.floor(sec / 60);
  const s = sec - m * 60;
  return `${String(m).padStart(2, "0")}:${s.toFixed(2).padStart(5, "0")}`;
}

function fmtTs(ts) {
  if (!ts) return "";
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString("ru-RU", { hour12: false }) + "." + String(d.getMilliseconds()).padStart(3, "0");
}

function fmtBytes(n) {
  if (n == null || !isFinite(n)) return "—";
  const abs = Math.abs(n);
  if (abs >= 1073741824) return `${(n / 1073741824).toFixed(2)} ГиБ`;
  if (abs >= 1048576) return `${(n / 1048576).toFixed(1)} МиБ`;
  if (abs >= 1024) return `${(n / 1024).toFixed(0)} КиБ`;
  return `${Math.round(n)} Б`;
}

function updateSpeedo(rootSel, pct, valueText, subText) {
  const root = rootSel.startsWith("#") ? $(rootSel) : document.getElementById(rootSel);
  if (!root) return;
  const p = Math.max(0, Math.min(100, Number.isFinite(pct) ? pct : 0));
  root.style.setProperty("--pct", String(p));
  const valEl = root.querySelector(".speedo-value");
  const subEl = root.querySelector(".speedo-sub");
  if (valEl) valEl.textContent = valueText;
  if (subEl) subEl.textContent = subText;
}

function switchPage(page) {
  state.currentPage = page;
  const isResources = page === "resources";
  $("#nav-monitor").classList.toggle("nav-pill-active", !isResources);
  $("#nav-resources").classList.toggle("nav-pill-active", isResources);
  $("#page-monitor").classList.toggle("hidden", isResources);
  $("#page-resources").classList.toggle("hidden", !isResources);
  $("#page-monitor").classList.toggle("page-panel-active", !isResources);
  $("#page-resources").classList.toggle("page-panel-active", isResources);
  if (isResources) {
    refreshMetrics();
    if (state.metricsTimer == null) state.metricsTimer = setInterval(refreshMetrics, 2500);
  } else if (state.metricsTimer != null) {
    clearInterval(state.metricsTimer);
    state.metricsTimer = null;
  }
}

async function api(method, path, body) {
  const opts = { method, headers: { "Content-Type": "application/json" } };
  if (body !== undefined) opts.body = JSON.stringify(body);
  let r;
  try {
    r = await fetch(path, opts);
  } catch (e) {
    throw new Error("Сервер недоступен. Проверьте, что python run.py запущен.");
  }
  if (!r.ok) {
    const text = await r.text();
    throw new Error(`${r.status}: ${text}`);
  }
  return r.json();
}

function withStream(path, streamId = state.activeStreamId) {
  const hasQ = path.includes("?");
  return `${path}${hasQ ? "&" : "?"}stream_id=${encodeURIComponent(streamId)}`;
}

function applyTheme(theme) {
  state.theme = theme === "dark" ? "dark" : "light";
  document.body.classList.toggle("theme-dark", state.theme === "dark");
  const btn = $("#btn-theme");
  if (btn) btn.textContent = state.theme === "dark" ? "☀" : "🌙";
  try { localStorage.setItem("integra_theme", state.theme); } catch {}
}

function renderSkeletonList(root, rows = 4) {
  if (!root) return;
  root.innerHTML = "";
  for (let i = 0; i < rows; i++) {
    const line = document.createElement("div");
    line.className = "skeleton h-7 mb-2";
    root.appendChild(line);
  }
}

// ----------------------------------------------------------- опрос статуса

async function refreshInfo() {
  try {
    const info = await api("GET", withStream("/api/info"));
    applyInfo(info);
  } catch (e) {
    console.warn("info fail", e);
  }
}

async function refreshMetrics() {
  try {
    const data = await api("GET", withStream("/api/metrics"));
    applySystemMetrics(data.system || {});
    applyProcessAnalyticsMetrics(data.process || {}, data.pipeline || {});
    renderStreamsMemory(state.streams || []);
  } catch (e) {
    console.warn("metrics fail", e);
  }
}

function applySystemMetrics(system) {
  const cpu = Number(system.cpu_percent || 0);
  updateSpeedo("speedo-cpu", cpu, cpu.toFixed(0) + "%", "100%");

  const ramUsed = system.ram_used_bytes || 0;
  const ramTotal = system.ram_total_bytes || 0;
  const ramPct = Number(system.ram_percent || 0);
  updateSpeedo("speedo-ram", ramPct, fmtBytes(ramUsed), fmtBytes(ramTotal));

  const gpu = system.gpu || {};
  const gpuName = gpu.name || "GPU не обнаружена";
  $("#gpu-name").textContent = gpuName;
  const gpuPct = Number(gpu.util_percent || 0);
  const gpuUsed = fmtBytes(gpu.memory_used_bytes || 0);
  const gpuTotal = fmtBytes(gpu.memory_total_bytes || 0);
  updateSpeedo("speedo-gpu", gpuPct, gpuUsed, gpuTotal);

  renderProcessesTopRss(system.processes_top_rss || []);
}

function applyProcessAnalyticsMetrics(proc, pipe) {
  const thr = pipe.thresholds || {};
  const warnB = Number(thr.memory_warning_bytes || 805306368);
  const critB = Number(thr.memory_critical_bytes || 1006632960);
  const rss = proc.rss_analytics_sum_bytes;
  const ema = proc.rss_ema_bytes;
  const cudaR = proc.cuda_memory_reserved_bytes;
  const q = pipe.queues || {};
  const buf = pipe.buffers || {};
  const st = pipe.stats || {};
  const leg = $("#mem-chart-legend");
  if (leg) {
    const pipe = proc.rss_pipeline_bytes;
    const inf = proc.rss_inference_worker_bytes;
    const parts = [
      `Σ RSS ${fmtBytes(rss)}`,
      `EMA ${fmtBytes(ema)}`,
      pipe != null ? `пайплайн ${fmtBytes(pipe)}` : null,
      inf != null && inf > 0 ? `инференс ${fmtBytes(inf)}` : null,
      `пик(окно) ${fmtBytes(proc.rss_peak_recent_bytes)}`,
      `decode ${q.decode_size ?? "—"}/${q.decode_max ?? "—"}`,
      `render ${q.render_size ?? "—"}/${q.render_max ?? "—"}`,
      `e2e EMA ${(st.e2e_ms_ema || 0).toFixed(0)} мс`,
    ].filter(Boolean);
    if (cudaR != null) parts.push(`CUDA ${fmtBytes(cudaR)}`);
    if (buf.frame_pool_free != null) parts.push(`pool своб. ${buf.frame_pool_free}/${buf.frame_pool_slots || "—"}`);
    parts.push(`forensic ${buf.forensic_ring_len || 0}`);
    leg.textContent = parts.join(" · ");
  }
  drawMemChart(proc.rss_history || [], warnB, critB);
  renderMemoryBreakdown(proc.memory_breakdown || []);
}

function renderMemoryBreakdown(rows) {
  const el = $("#mem-breakdown");
  if (!el) return;
  el.replaceChildren();
  if (!rows.length) {
    el.textContent = "—";
    return;
  }
  for (const row of rows) {
    const block = document.createElement("div");
    block.className = "mb-1.5 border-b border-slate-100 pb-1 last:border-0";
    const line = document.createElement("div");
    line.className = "flex justify-between gap-2";
    const lab = document.createElement("span");
    lab.className = "text-slate-600 truncate";
    lab.textContent =
      row.label + (row.pid != null && row.kind === "process_rss" ? " · pid " + row.pid : "");
    const val = document.createElement("span");
    val.className = "text-slate-800 font-mono shrink-0";
    val.textContent = row.bytes != null ? fmtBytes(row.bytes) : "—";
    line.append(lab, val);
    block.appendChild(line);
    if (row.hint) {
      const hint = document.createElement("div");
      hint.className = "text-slate-400 text-[9px] mt-0.5";
      hint.textContent = row.hint;
      block.appendChild(hint);
    }
    el.appendChild(block);
  }
}

function renderProcessesTopRss(rows) {
  const el = $("#sys-top-procs");
  if (!el) return;
  el.replaceChildren();
  if (!rows.length) {
    el.textContent = "нет данных (psutil недоступен или ни один процесс не выше 80 MB RSS)";
    return;
  }
  for (const row of rows) {
    const line = document.createElement("div");
    line.className = "flex justify-between gap-2 mb-0.5 font-mono text-[9px]";
    const lab = document.createElement("span");
    lab.className = "text-slate-600 truncate";
    lab.textContent = (row.name || "?") + " · " + (row.pid ?? "—");
    const val = document.createElement("span");
    val.className = "text-slate-800 shrink-0";
    val.textContent = fmtBytes(row.rss_bytes || 0);
    line.append(lab, val);
    el.appendChild(line);
  }
}

function drawMemChart(hist, warnB, critB) {
  const cv = $("#mem-chart");
  if (!cv || !cv.getContext) return;
  const ctx = cv.getContext("2d");
  const padL = 44;
  const w = cv.clientWidth || cv.width;
  const h = 120;
  if (cv.width !== w) cv.width = w;
  if (cv.height !== h) cv.height = h;
  ctx.fillStyle = "#f8fafc";
  ctx.fillRect(0, 0, w, h);
  if (!hist.length) return;
  const totalPts = hist.map((p) => Number(p.rss_total_bytes ?? p.rss_ema_bytes) || 0);
  const pipePts = hist.map((p) => Number(p.rss_pipeline_bytes ?? p.rss_total_bytes ?? p.rss_ema_bytes) || 0);
  const inferPts = hist.map((p) => (p.rss_inference_bytes != null ? Number(p.rss_inference_bytes) || 0 : 0));
  const hasInfer = inferPts.some((v) => v > 1e5);
  const maxV = Math.max(critB * 1.02, ...totalPts, ...pipePts, ...(hasInfer ? inferPts : [0]), 1);
  const t0 = hist[0].t;
  const t1 = hist[hist.length - 1].t;
  const maxT = Math.max(t1 - t0, 1e-6);
  const plotW = w - padL - 4;
  const lineY = (bytes) => h - 8 - (bytes / maxV) * (h - 16);
  ctx.fillStyle = "#94a3b8";
  ctx.font = "9px sans-serif";
  ctx.textAlign = "right";
  for (const frac of [0, 0.5, 1]) {
    const b = maxV * frac;
    const y = lineY(b);
    ctx.fillText(fmtBytes(b), padL - 4, y + 3);
  }
  ctx.textAlign = "left";
  ctx.strokeStyle = "#e2e8f0";
  for (let i = 0; i <= 4; i++) {
    const y = 8 + (i / 4) * (h - 16);
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }
  ctx.strokeStyle = "#fbbf24";
  ctx.setLineDash([5, 5]);
  ctx.beginPath();
  ctx.moveTo(padL, lineY(warnB));
  ctx.lineTo(w, lineY(warnB));
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.strokeStyle = "#f87171";
  ctx.beginPath();
  ctx.moveTo(padL, lineY(critB));
  ctx.lineTo(w, lineY(critB));
  ctx.stroke();

  const strokeSeries = (pts, color, width, dash) => {
    ctx.setLineDash(dash || []);
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    hist.forEach((pt, i) => {
      const x = padL + ((pt.t - t0) / maxT) * (plotW - 2) + 1;
      const y = lineY(pts[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.lineWidth = 1;
  };

  strokeSeries(pipePts, "#1d4ed8", 1.8, []);
  if (hasInfer) strokeSeries(inferPts, "#059669", 1.5, [4, 3]);
  else strokeSeries(totalPts, "#64748b", 1, [2, 2]);

  ctx.font = "9px sans-serif";
  ctx.textAlign = "right";
  if (hasInfer) {
    ctx.fillStyle = "#1d4ed8";
    ctx.fillText("— пайплайн (декод, FSM, API)", w - 4, 12);
    ctx.fillStyle = "#059669";
    ctx.fillText("— инференс (YOLO+трекер)", w - 4, 24);
  } else {
    ctx.fillStyle = "#64748b";
    ctx.fillText("— суммарный RSS (YOLO в этом процессе)", w - 4, 12);
  }
  ctx.textAlign = "left";
}

function applyInfo(info) {
  state.loaded = info.loaded;
  state.playing = info.playing;
  state.totalFrames = info.frame_count;
  state.fps = info.fps;
  $("#no-video").style.display = info.loaded ? "none" : "flex";

  const cur = info.current_frame;
  if (!state.seekDragging) {
    $("#seek").max = String(Math.max(0, info.frame_count - 1));
    $("#seek").value = String(cur);
  }
  $("#time-cur").textContent = fmtTime(info.current_sec);
  $("#time-total").textContent = fmtTime(info.duration_sec);
  $("#frame-info").textContent = `кадр ${cur} / ${info.frame_count}`;
  $("#fps-info").textContent = `${info.fps?.toFixed(2) ?? "—"} fps`;
  $("#resolution-info").textContent = `${info.width || "—"} × ${info.height || "—"}`;

  const s = info.stats || {};
  $("#hud-decode").textContent = `декод ${(s.decode_fps || 0).toFixed(0)} fps`;
  $("#hud-render").textContent = `рендер ${(s.render_fps || 0).toFixed(0)} fps`;
  $("#hud-inf").textContent = `инф ${(s.inference_ms_avg || 0).toFixed(1)} мс`;
  $("#hud-events").textContent = `событий ${s.events || 0}`;
  const pipeFps = s.pipeline_fps != null ? s.pipeline_fps.toFixed(0) : "—";
  $("#dropped-info").textContent = `пропуск decode ${s.dropped_decode || 0} · render ${s.dropped_render || 0} · пайплайн ${pipeFps} fps`;

  // Синхронизируем локальную карту тревог с актуальным snapshot треков.
  const liveAlarms = new Set();
  for (const t of info.tracks || []) {
    if (t.alarm) {
      liveAlarms.add(t.id);
      if (!state.alarms.has(t.id)) {
        state.alarms.set(t.id, { id: t.id, cls: t.cls, state: t.state, bbox: t.bbox });
      } else {
        state.alarms.get(t.id).state = t.state;
      }
    }
  }
  // Удаляем тревоги, чьи треки уже исчезли из анализатора.
  for (const id of [...state.alarms.keys()]) {
    if (!liveAlarms.has(id)) state.alarms.delete(id);
  }
  renderAlarms();
}

async function refreshStreams() {
  try {
    const data = await api("GET", "/api/streams");
    state.streams = data.streams || [];
    if (!state.streams.some((s) => s.stream_id === state.activeStreamId)) {
      state.activeStreamId = "main";
    }
    renderStreamsTabs();
    renderStreamsMemory(state.streams);
  } catch (e) {
    console.warn("streams fail", e);
  }
}

function renderStreamsTabs() {
  const root = $("#streams-tabs");
  if (!root) return;
  root.innerHTML = "";
  if (!state.streams.length) {
    root.innerHTML = '<div class="text-xs text-slate-400">Потоки не созданы</div>';
    return;
  }
  for (const s of state.streams) {
    const btn = document.createElement("button");
    const active = s.stream_id === state.activeStreamId;
    btn.className = active ? "stream-chip stream-chip-active" : "stream-chip";
    const baseName = s.stream_id;
    const status = s.playing ? "▶" : "⏸";
    btn.textContent = `${status} ${baseName}`;
    btn.onclick = () => selectStream(s.stream_id);
    root.appendChild(btn);
  }
}

function renderStreamsMemory(streams) {
  const root = $("#streams-mem-list");
  if (!root) return;
  root.replaceChildren();
  if (!streams.length) {
    root.textContent = "—";
    return;
  }
  for (const s of streams) {
    const row = document.createElement("div");
    row.className = "flex items-center justify-between gap-2 mb-0.5";
    const label = document.createElement("span");
    label.className = "truncate";
    label.textContent = `${s.stream_id}${s.stream_id === state.activeStreamId ? " (активный)" : ""}`;
    const total = s.memory?.estimated_total_bytes;
    const worker = s.memory?.worker_rss_bytes;
    const val = document.createElement("span");
    val.className = "font-mono text-slate-800";
    val.textContent = `${fmtBytes(total)}${worker ? ` · worker ${fmtBytes(worker)}` : ""}`;
    row.append(label, val);
    root.appendChild(row);
  }
}

async function selectStream(streamId) {
  state.activeStreamId = streamId;
  state.alarms.clear();
  renderAlarms();
  renderStreamsTabs();
  await refreshInfo();
  await refreshEvents();
  await refreshMetrics();
  const v = $("#video");
  attachVideoStream(streamId);
}

function renderAlarms() {
  const root = $("#alarms");
  root.innerHTML = "";
  const count = state.alarms.size;
  const cnt = $("#alarm-count");
  cnt.textContent = String(count);
  cnt.className = count > 0 ? "badge badge-danger" : "badge badge-neutral";
  $("#alarm-overlay").classList.toggle("hidden", count === 0);
  if (count === 0) {
    root.innerHTML = '<div class="text-xs text-slate-400">Нет активных тревог.</div>';
    return;
  }
  for (const a of state.alarms.values()) {
    const el = document.createElement("div");
    const badgeClass = a.state === "alarm_disappeared" ? "badge badge-warn" : "badge badge-danger";
    el.className = "alarm-card alarm-active";
    el.innerHTML = `
      <div>
        <div class="text-sm font-medium text-slate-900">#${a.id} · ${a.cls}</div>
        <div class="text-[11px] text-slate-500">${a.state === "alarm_disappeared" ? "предмет пропал" : "обнаружен предмет"}</div>
      </div>
      <span class="${badgeClass}">${a.state === "alarm_disappeared" ? "ПРОПАЛ" : "ОБНАРУЖЕН"}</span>
    `;
    root.appendChild(el);
  }
}

// ----------------------------------------------------------- журнал событий

async function refreshEvents() {
  try {
    renderSkeletonList($("#events"), 5);
    const data = await api("GET", withStream("/api/events?limit=200"));
    renderEvents(data.events || []);
  } catch (e) { console.warn(e); }
}

function renderEvents(events) {
  const root = $("#events");
  root.innerHTML = "";
  if (!events.length) {
    root.innerHTML = '<div class="text-xs text-slate-400">Журнал пуст.</div>';
    return;
  }
  for (const ev of events) {
    if (ev.type !== "abandoned" && ev.type !== "disappeared") continue;
    const ui = EVENT_UI[ev.type] || { title: ev.type, desc: "" };
    const card = document.createElement("div");
    card.className = `event-card event-${ev.type}`;
    const img = ev.snapshot_path
      ? `<a href="${ev.snapshot_path}" target="_blank" rel="noopener noreferrer" title="Открыть снимок">
           <img src="${ev.snapshot_path}" alt="снимок события"/>
         </a>`
      : '<div class="placeholder">нет</div>';
    const note = ev.note ? `<div class="text-[11px] text-slate-500 mt-1">${ev.note}</div>` : "";
    card.innerHTML = `
      ${img}
      <div class="flex-1 min-w-0">
        <div class="flex items-center gap-2 text-xs">
          <span class="ev-type">${ui.title}</span>
          <span class="text-slate-500">#${ev.track_id ?? "—"}</span>
          <span class="text-slate-700 font-medium">${ev.cls_name ?? ""}</span>
          ${ev.confidence != null ? `<span class="text-slate-400">${(ev.confidence * 100).toFixed(0)}%</span>` : ""}
        </div>
        <div class="text-[11px] text-slate-500 mt-0.5">${fmtTs(ev.ts)} · ${(ev.video_pos_ms / 1000).toFixed(2)}s</div>
        <div class="text-[11px] text-slate-600 mt-0.5">${ui.desc}</div>
        ${note}
      </div>
    `;
    root.appendChild(card);
  }
}

// ----------------------------------------------------------- список видеофайлов

async function refreshFiles() {
  const root = $("#files-list");
  renderSkeletonList(root, 4);
  const data = await api("GET", "/api/files");
  root.innerHTML = "";
  if (!data.files.length) {
    root.innerHTML = `<div class="col-span-2 text-xs text-slate-400">Положите .mkv в ${data.data_dir}</div>`;
    return;
  }
  for (const f of data.files) {
    const btn = document.createElement("button");
    btn.className = "file-tile truncate";
    btn.innerHTML = `<div class="name truncate">${f.name}</div><div class="meta">${f.size_mb} MB</div>`;
    btn.onclick = () => openVideo(f.path);
    root.appendChild(btn);
  }
}

async function openVideo(path) {
  try {
    await api("POST", "/api/open", { path, stream_id: state.activeStreamId });
    await refreshStreams();
    await refreshInfo();
    await refreshEvents();
    // Принудительно переподключаем MJPEG-поток через смену query-параметра.
    attachVideoStream(state.activeStreamId);
  } catch (e) { alert("Не удалось открыть: " + e.message); }
}

async function openVideoFromSystemPicker(file) {
  if (!file) return;
  $("#picked-file-name").textContent = `Загрузка: ${file.name}`;
  const fd = new FormData();
  fd.append("file", file, file.name);
  try {
    let r;
    try {
      r = await fetch("/api/upload_video", { method: "POST", body: fd });
    } catch (e) {
      throw new Error("Сервер недоступен. Проверьте, что python run.py запущен.");
    }
    if (!r.ok) {
      const t = await r.text();
      throw new Error(`${r.status}: ${t}`);
    }
    const data = await r.json();
    $("#picked-file-name").textContent = `Загружен: ${data.name}`;
    await refreshFiles();
    await openVideo(data.path);
  } catch (e) {
    $("#picked-file-name").textContent = "Ошибка загрузки";
    alert("Не удалось загрузить файл: " + e.message);
  }
}

// ----------------------------------------------------------- настройки

async function loadSettings() {
  state.settings = await api("GET", "/api/settings");
  applySettingsToForm(state.settings);
}

function applySettingsToForm(cfg) {
  for (const inp of $$("#settings-form [data-section]")) {
    const section = inp.dataset.section;
    const key = inp.dataset.key;
    const val = cfg?.[section]?.[key];
    if (val === undefined) continue;
    if (inp.type === "checkbox") inp.checked = !!val;
    else inp.value = val;
  }
}

async function submitSettings(ev) {
  ev.preventDefault();
  // Собираем PATCH только из полей формы, оставляя пустые значения без отправки.
  const patch = { model: {}, pipeline: {}, analyzer: {}, ui: {} };
  for (const inp of $$("#settings-form [data-section]")) {
    const section = inp.dataset.section;
    const key = inp.dataset.key;
    let val;
    if (inp.type === "checkbox") val = inp.checked;
    else if (inp.type === "number") val = inp.value === "" ? null : Number(inp.value);
    else val = inp.value;
    if (val !== null && val !== "") patch[section][key] = val;
  }
  try {
    state.settings = await api("PUT", "/api/settings", patch);
    flash("Настройки применены");
  } catch (e) { alert("Ошибка: " + e.message); }
}

function flash(text, kind = "success") {
  const el = document.createElement("div");
  el.className = "toast " + (kind === "error" ? "toast-error" : "toast-success");
  el.textContent = text;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 1800);
}

// ----------------------------------------------------------- WebSocket

function connectWs() {
  const url = (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws";
  const ws = new WebSocket(url);
  state.ws = ws;
  ws.onopen = () => {
    $("#hud-conn").textContent = "в сети";
    $("#hud-conn").className = "metric metric-good";
  };
  ws.onclose = () => {
    $("#hud-conn").textContent = "не в сети";
    $("#hud-conn").className = "metric metric-bad";
    // Автопереподключение, чтобы UI сам восстанавливался после обрывов.
    setTimeout(connectWs, 1500);
  };
  ws.onerror = () => ws.close();
  ws.onmessage = (m) => {
    let msg;
    try { msg = JSON.parse(m.data); } catch { return; }
    if (msg.type === "status") {
      if (!msg.stream_id || msg.stream_id === state.activeStreamId) applyInfo(msg.info);
    } else if (msg.type === "hello") {
      if (!msg.stream_id || msg.stream_id === state.activeStreamId) applyInfo(msg.info);
    }
    else if (msg.type === "event") onEventMessage(msg);
  };
}

function onEventMessage(ev) {
  if (ev.stream_id && ev.stream_id !== state.activeStreamId) return;
  refreshEvents();
  if (ev.type === "abandoned" || ev.type === "disappeared") {
    if (state.settings?.ui?.alarm_sound) {
      try { $("#alarm-sound").currentTime = 0; $("#alarm-sound").play(); } catch {}
    }
  }
}

// ----------------------------------------------------------- привязка обработчиков UI

function wireControls() {
  $("#btn-play").onclick = () => api("POST", withStream("/api/play")).then(refreshInfo);
  $("#btn-pause").onclick = () => api("POST", withStream("/api/pause")).then(refreshInfo);

  const seek = $("#seek");
  seek.addEventListener("mousedown", () => state.seekDragging = true);
  seek.addEventListener("touchstart", () => state.seekDragging = true);
  const release = () => {
    if (!state.seekDragging) return;
    state.seekDragging = false;
    // Отправляем seek один раз при отпускании ползунка.
    api("POST", withStream("/api/seek"), { frame: Number(seek.value) }).then(refreshInfo);
  };
  seek.addEventListener("mouseup", release);
  seek.addEventListener("touchend", release);
  seek.addEventListener("change", () => {
    if (!state.seekDragging) {
      api("POST", withStream("/api/seek"), { frame: Number(seek.value) }).then(refreshInfo);
    }
  });

  $("#btn-refresh-files").onclick = refreshFiles;
  $("#btn-refresh-streams").onclick = refreshStreams;
  $("#btn-add-stream").onclick = async () => {
    const sid = prompt("Введите ID потока (латиница/цифры/_/-):", "");
    if (!sid) return;
    await api("POST", "/api/streams", { stream_id: sid });
    await refreshStreams();
    await selectStream(sid);
  };
  $("#btn-open-path").onclick = () => {
    const p = $("#custom-path").value.trim();
    if (p) openVideo(p);
  };
  $("#btn-open-system-file").onclick = () => $("#video-file-picker").click();
  $("#video-file-picker").addEventListener("change", async (ev) => {
    const file = ev.target.files && ev.target.files[0];
    await openVideoFromSystemPicker(file);
    ev.target.value = "";
  });
  $("#btn-refresh-events").onclick = refreshEvents;
  $("#btn-clear-events").onclick = async () => {
    if (!confirm("Очистить журнал?")) return;
    await api("DELETE", withStream("/api/events"));
    refreshEvents();
  };

  $("#btn-toggle-settings").onclick = () => {
    const f = $("#settings-form");
    f.classList.toggle("hidden");
    $("#settings-arrow").textContent = f.classList.contains("hidden") ? "▾" : "▴";
  };
  $("#btn-reload-settings").onclick = loadSettings;
  $("#settings-form").addEventListener("submit", submitSettings);
  $("#nav-monitor").onclick = () => switchPage("monitor");
  $("#nav-resources").onclick = () => switchPage("resources");
  $("#btn-theme").onclick = () => applyTheme(state.theme === "dark" ? "light" : "dark");
}

async function main() {
  try {
    const savedTheme = localStorage.getItem("integra_theme");
    if (savedTheme) applyTheme(savedTheme);
    else applyTheme("light");
  } catch {
    applyTheme("light");
  }
  wireControls();
  await loadSettings();
  await refreshStreams();
  await refreshFiles();
  await refreshEvents();
  await refreshInfo();
  await refreshMetrics();
  switchPage("monitor");
  const v = $("#video");
  attachVideoStream(state.activeStreamId);
  connectWs();
  state.streamsTimer = setInterval(refreshStreams, 4000);
}

function attachVideoStream(streamId) {
  const v = $("#video");
  if (!v) return;
  // Reset any previous handlers/intervals
  if (v._snapshotPoll) {
    clearInterval(v._snapshotPoll);
    v._snapshotPoll = null;
  }
  v.onerror = () => {
    // MJPEG failed — fallback to polling single-image snapshot.
    if (v._snapshotPoll) return;
    v._snapshotPoll = setInterval(() => {
      v.src = withStream("/video_snapshot?ts=" + Date.now(), streamId);
    }, 300);
  };
  // Try MJPEG first (cache-buster)
  v.src = withStream("/video_feed?ts=" + Date.now(), streamId);
}

main().catch(console.error);
