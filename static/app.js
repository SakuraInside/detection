/* Front-end Integra-LOST.
   Сборка не нужна: чистый ES2020 в браузере.
   Мультипоток: до MAX_STREAMS панелей (камеры/файлы) работают ОДНОВРЕМЕННО,
   каждая со своим видео/таймлайном; журнал событий и тревоги — общие с пометкой потока. */

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const MAX_STREAMS = 4;

const state = {
  currentPage: "monitor",
  // stream_id -> { info, panel:{...}, seekDragging }
  streams: new Map(),
  // ключ `${stream_id}:${track_id}` -> { stream_id, id, cls, state }
  alarms: new Map(),
  ws: null,
  settings: null,
  metricsTimer: null,
  streamsTimer: null,
  theme: "light",
};

const EVENT_UI = {
  person_interaction: { title: "Взаимодействие", desc: "Человек подошёл к объекту сцены." },
  object_left: { title: "Предмет оставлен", desc: "Объект оставлен, владелец отошёл." },
  object_unattended: { title: "Бесхозный предмет", desc: "Предмет без владельца дольше порога." },
  object_removed: { title: "Предмет забрали", desc: "Объект исчез после взаимодействия с человеком." },
  object_missing: { title: "Предмет пропал", desc: "Ранее наблюдавшийся объект исчез из кадра." },
  system_error: { title: "Ошибка источника", desc: "Поток/файл не открылся." },
};

const JOURNAL_EVENT_TYPES = new Set([
  "object_left", "object_unattended", "object_removed", "object_missing", "system_error",
]);

const ALARM_EVENT_TYPES = new Set([
  "object_unattended", "object_removed", "object_missing",
]);

const ALARM_STATE_UI = {
  alarm_unattended: { label: "бесхозный предмет", badge: "badge badge-danger", tag: "БЕСХОЗНЫЙ" },
  alarm_removed: { label: "предмет забрали", badge: "badge badge-warn", tag: "ЗАБРАЛИ" },
  alarm_missing: { label: "предмет пропал", badge: "badge badge-warn", tag: "ПРОПАЛ" },
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

function withStream(path, streamId) {
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

// ----------------------------------------------------------- потоки и панели

/** Свободный cam-id (cam1..camN), которого ещё нет среди потоков. */
function nextStreamId() {
  for (let i = 1; i <= MAX_STREAMS; i++) {
    const id = "cam" + i;
    if (!state.streams.has(id)) return id;
  }
  return null; // лимит достигнут
}

/** Создать (или вернуть) запись потока + DOM-панель. */
function ensureStream(streamId) {
  let s = state.streams.get(streamId);
  if (s) return s;
  const panel = buildPanel(streamId);
  s = { info: null, panel, seekDragging: false };
  state.streams.set(streamId, s);
  $("#streams-grid").appendChild(panel.root);
  syncGridLayout();
  renderTargetSelect();
  return s;
}

function removeStream(streamId) {
  const s = state.streams.get(streamId);
  if (!s) return;
  s.panel.root.remove();
  state.streams.delete(streamId);
  // убрать тревоги этого потока
  for (const key of [...state.alarms.keys()]) {
    if (key.startsWith(streamId + ":")) state.alarms.delete(key);
  }
  renderAlarms();
  syncGridLayout();
  renderTargetSelect();
}

/** Раскладка сетки: 1 поток — широкая, 2+ — по два столбца. */
function syncGridLayout() {
  const grid = $("#streams-grid");
  const n = state.streams.size;
  grid.classList.toggle("cols-1", n <= 1);
  grid.classList.toggle("cols-2", n >= 2);
  $("#streams-empty").style.display = n === 0 ? "block" : "none";
  $("#streams-count").textContent = n ? `· ${n}/${MAX_STREAMS}` : "";
}

/** Построить DOM-панель потока с собственными контролами. */
function buildPanel(streamId) {
  const root = document.createElement("div");
  root.className = "stream-panel card-animated";
  root.dataset.stream = streamId;
  root.innerHTML = `
    <div class="panel-head">
      <span class="panel-dot"></span>
      <span class="panel-title" title="${streamId}">${streamId}</span>
      <span class="panel-src text-slate-400 truncate"></span>
      <button class="panel-close" title="Закрыть поток">✕</button>
    </div>
    <div class="panel-video-wrap">
      <img class="panel-video" alt="${streamId}"/>
      <div class="panel-alarm-ring hidden"></div>
      <div class="panel-novideo">Нет видео</div>
    </div>
    <div class="panel-controls">
      <button class="panel-play btn-icon" title="Воспроизведение">▶</button>
      <button class="panel-pause btn-icon" title="Пауза">⏸</button>
      <input class="panel-seek" type="range" min="0" max="0" value="0" step="1"/>
      <span class="panel-meta">0 / 0 · — fps</span>
    </div>
  `;
  const panel = {
    root,
    img: root.querySelector(".panel-video"),
    ring: root.querySelector(".panel-alarm-ring"),
    novideo: root.querySelector(".panel-novideo"),
    title: root.querySelector(".panel-title"),
    src: root.querySelector(".panel-src"),
    seek: root.querySelector(".panel-seek"),
    meta: root.querySelector(".panel-meta"),
    playBtn: root.querySelector(".panel-play"),
    pauseBtn: root.querySelector(".panel-pause"),
  };

  panel.playBtn.onclick = () =>
    api("POST", withStream("/api/play", streamId)).catch((e) => console.warn(e));
  panel.pauseBtn.onclick = () =>
    api("POST", withStream("/api/pause", streamId)).catch((e) => console.warn(e));
  panel.close = root.querySelector(".panel-close");
  panel.close.onclick = async () => {
    try { await api("POST", "/api/close_stream", { stream_id: streamId }); } catch {}
    removeStream(streamId);
  };

  const s = () => state.streams.get(streamId);
  panel.seek.addEventListener("mousedown", () => { if (s()) s().seekDragging = true; });
  panel.seek.addEventListener("touchstart", () => { if (s()) s().seekDragging = true; });
  const release = () => {
    const st = s();
    if (!st || !st.seekDragging) return;
    st.seekDragging = false;
    api("POST", withStream("/api/seek", streamId), { frame: Number(panel.seek.value) })
      .catch((e) => console.warn(e));
  };
  panel.seek.addEventListener("mouseup", release);
  panel.seek.addEventListener("touchend", release);
  panel.seek.addEventListener("change", () => {
    const st = s();
    if (st && !st.seekDragging) {
      api("POST", withStream("/api/seek", streamId), { frame: Number(panel.seek.value) })
        .catch((e) => console.warn(e));
    }
  });

  attachVideoStream(streamId, panel);
  return panel;
}

/** Применить info одного потока к его панели + синхронизировать тревоги. */
function applyInfo(streamId, info) {
  if (!info) return;
  const s = ensureStream(streamId);
  s.info = info;
  const p = s.panel;

  p.novideo.style.display = info.loaded ? "none" : "flex";
  if (info.video_path) {
    const base = String(info.video_path).split(/[\\/]/).pop();
    p.src.textContent = base ? "· " + base : "";
    p.title.title = info.video_path;
  }
  p.root.classList.toggle("panel-paused", info.loaded && !info.playing);

  const cur = info.current_frame || 0;
  if (!s.seekDragging) {
    p.seek.max = String(Math.max(0, (info.frame_count || 1) - 1));
    p.seek.value = String(cur);
  }
  const st = info.stats || {};
  const dec = (st.decode_fps || 0).toFixed(0);
  p.meta.textContent =
    `${cur} / ${info.frame_count || 0} · ${dec} fps · ${info.width || "—"}×${info.height || "—"}`;

  // тревоги этого потока из snapshot треков
  const live = new Set();
  for (const t of info.tracks || []) {
    if (t.alarm) {
      const key = streamId + ":" + t.id;
      live.add(key);
      if (!state.alarms.has(key)) {
        state.alarms.set(key, { stream_id: streamId, id: t.id, cls: t.cls, state: t.state });
      } else {
        state.alarms.get(key).state = t.state;
      }
    }
  }
  for (const key of [...state.alarms.keys()]) {
    if (key.startsWith(streamId + ":") && !live.has(key)) state.alarms.delete(key);
  }
  const hasAlarm = [...state.alarms.keys()].some((k) => k.startsWith(streamId + ":"));
  p.ring.classList.toggle("hidden", !hasAlarm);
  renderAlarms();
  updateHud();
}

/** Сводный HUD в шапке (сумма по потокам). */
function updateHud() {
  let dec = 0, inf = 0, ev = 0, n = 0;
  for (const s of state.streams.values()) {
    const st = s.info?.stats; if (!st) continue;
    dec += st.decode_fps || 0; inf += st.inference_ms_avg || 0; ev += st.events || 0; n++;
  }
  $("#hud-decode").textContent = `декод Σ${dec.toFixed(0)} fps`;
  $("#hud-render").textContent = `потоков ${state.streams.size}`;
  $("#hud-inf").textContent = `инф ${n ? (inf / n).toFixed(0) : 0} мс`;
  $("#hud-events").textContent = `событий ${ev}`;
  for (const el of ["#hud-decode", "#hud-render", "#hud-inf", "#hud-events", "#hud-conn"]) {
    $(el).classList.remove("hidden");
  }
}

/** Синхронизировать панели с серверным списком потоков (GET /api/streams). */
async function refreshStreams() {
  try {
    const data = await api("GET", "/api/streams");
    const serverStreams = data.streams || [];
    const seen = new Set();
    for (const srv of serverStreams) {
      seen.add(srv.stream_id);
      ensureStream(srv.stream_id);
    }
    // потоки, исчезнувшие на сервере — убрать (если не локально-новые без open)
    for (const id of [...state.streams.keys()]) {
      if (!seen.has(id) && state.streams.get(id).info) removeStream(id);
    }
    renderStreamsMemory(serverStreams);
    renderTargetSelect();
  } catch (e) {
    console.warn("streams fail", e);
  }
}

function renderTargetSelect() {
  const sel = $("#target-stream");
  if (!sel) return;
  const prev = sel.value;
  sel.innerHTML = "";
  const full = state.streams.size >= MAX_STREAMS;
  const optNew = document.createElement("option");
  optNew.value = "__new__";
  optNew.textContent = full ? "Лимит потоков" : "Новый поток";
  optNew.disabled = full;
  sel.appendChild(optNew);
  for (const id of [...state.streams.keys()].sort()) {
    const o = document.createElement("option");
    o.value = id;
    o.textContent = id + " (заменить)";
    sel.appendChild(o);
  }
  if ([...sel.options].some((o) => o.value === prev)) sel.value = prev;
  else sel.value = full && state.streams.size ? [...state.streams.keys()].sort()[0] : "__new__";
}

function renderStreamsMemory(streams) {
  const root = $("#streams-mem-list");
  if (!root) return;
  root.replaceChildren();
  if (!streams || !streams.length) {
    root.textContent = "—";
    return;
  }
  for (const s of streams) {
    const row = document.createElement("div");
    row.className = "flex items-center justify-between gap-2 mb-0.5";
    const label = document.createElement("span");
    label.className = "truncate";
    label.textContent = `${s.stream_id}${s.playing ? " ▶" : ""}`;
    const val = document.createElement("span");
    val.className = "font-mono text-slate-800";
    val.textContent = s.video_path ? String(s.video_path).split(/[\\/]/).pop() : "—";
    row.append(label, val);
    root.appendChild(row);
  }
}

// ----------------------------------------------------------- тревоги (общие, с пометкой потока)

function renderAlarms() {
  const root = $("#alarms");
  root.innerHTML = "";
  const count = state.alarms.size;
  const cnt = $("#alarm-count");
  cnt.textContent = String(count);
  cnt.className = count > 0 ? "badge badge-danger" : "badge badge-neutral";
  if (count === 0) {
    root.innerHTML = '<div class="text-xs text-slate-400">Нет активных тревог.</div>';
    return;
  }
  for (const a of state.alarms.values()) {
    const el = document.createElement("div");
    const ui = ALARM_STATE_UI[a.state] || ALARM_STATE_UI.alarm_unattended;
    el.className = "alarm-card alarm-active";
    el.innerHTML = `
      <div>
        <div class="text-sm font-medium text-slate-900">#${a.id} · ${a.cls}</div>
        <div class="text-[11px] text-slate-500"><span class="stream-tag">${a.stream_id}</span> ${ui.label}</div>
      </div>
      <span class="${ui.badge}">${ui.tag}</span>
    `;
    root.appendChild(el);
  }
}

// ----------------------------------------------------------- журнал событий (общий)

async function refreshEvents() {
  try {
    renderSkeletonList($("#events"), 5);
    const data = await api("GET", "/api/events?limit=200");
    renderEvents(data.events || []);
  } catch (e) { console.warn(e); }
}

function renderEvents(events) {
  const root = $("#events");
  root.innerHTML = "";
  const shown = events.filter((ev) => JOURNAL_EVENT_TYPES.has(ev.type));
  if (!shown.length) {
    root.innerHTML = '<div class="text-xs text-slate-400">Журнал пуст.</div>';
    return;
  }
  for (const ev of shown) {
    const ui = EVENT_UI[ev.type] || { title: ev.type, desc: "" };
    const card = document.createElement("div");
    card.className = `event-card event-${ev.type}`;
    const img = ev.snapshot_path
      ? `<a href="${ev.snapshot_path}" target="_blank" rel="noopener noreferrer" title="Открыть снимок">
           <img src="${ev.snapshot_path}" alt="снимок события"/>
         </a>`
      : '<div class="placeholder">нет</div>';
    const note = ev.note ? `<div class="text-[11px] text-slate-500 mt-1">${ev.note}</div>` : "";
    const tag = ev.stream_id ? `<span class="stream-tag">${ev.stream_id}</span>` : "";
    card.innerHTML = `
      ${img}
      <div class="flex-1 min-w-0">
        <div class="flex items-center gap-2 text-xs">
          ${tag}
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

// ----------------------------------------------------------- список видеофайлов / открытие

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

/** Выбрать целевой stream_id из селектора (или новый свободный слот). */
function resolveTargetStream() {
  const sel = $("#target-stream");
  const v = sel ? sel.value : "__new__";
  if (v && v !== "__new__") return v;
  const id = nextStreamId();
  if (!id) {
    alert(`Достигнут лимит потоков (${MAX_STREAMS}). Закройте один, чтобы открыть новый.`);
    return null;
  }
  return id;
}

async function openVideo(path, targetStreamId) {
  const streamId = targetStreamId || resolveTargetStream();
  if (!streamId) return;
  try {
    ensureStream(streamId); // панель сразу, ещё до ответа
    await api("POST", "/api/open", { path, stream_id: streamId });
    attachVideoStream(streamId);
    await refreshStreams();
    await refreshEvents();
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

/** Привязать MJPEG-поток к панели (с fallback на polling снапшота). */
function attachVideoStream(streamId, panelArg) {
  const panel = panelArg || state.streams.get(streamId)?.panel;
  if (!panel) return;
  const v = panel.img;
  if (v._snapshotPoll) { clearInterval(v._snapshotPoll); v._snapshotPoll = null; }
  v.onerror = () => {
    if (v._snapshotPoll) return;
    v._snapshotPoll = setInterval(() => {
      v.src = withStream("/video_snapshot?ts=" + Date.now(), streamId);
    }, 300);
  };
  v.src = withStream("/video_feed?ts=" + Date.now(), streamId);
}

// ----------------------------------------------------------- метрики ресурсов

async function refreshMetrics() {
  try {
    // метрики системные одни на всех; берём по первому потоку (или без stream_id).
    const firstId = [...state.streams.keys()][0];
    const path = firstId ? withStream("/api/metrics", firstId) : "/api/metrics";
    const data = await api("GET", path);
    applySystemMetrics(data.system || {});
    applyProcessAnalyticsMetrics(data.process || {}, data.pipeline || {});
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
    const pipeB = proc.rss_pipeline_bytes;
    const inf = proc.rss_inference_worker_bytes;
    const parts = [
      `Σ RSS ${fmtBytes(rss)}`,
      `EMA ${fmtBytes(ema)}`,
      pipeB != null ? `пайплайн ${fmtBytes(pipeB)}` : null,
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
    setTimeout(connectWs, 1500);
  };
  ws.onerror = () => ws.close();
  ws.onmessage = (m) => {
    let msg;
    try { msg = JSON.parse(m.data); } catch { return; }
    if (msg.type === "status" || msg.type === "hello") {
      if (msg.stream_id) applyInfo(msg.stream_id, msg.info);
    } else if (msg.type === "event") {
      onEventMessage(msg.event || msg);
    }
  };
}

function onEventMessage(ev) {
  refreshEvents();
  if (ALARM_EVENT_TYPES.has(ev.type) && state.settings?.ui?.alarm_sound) {
    try { $("#alarm-sound").currentTime = 0; $("#alarm-sound").play(); } catch {}
  }
}

// ----------------------------------------------------------- привязка обработчиков UI

function wireControls() {
  $("#btn-refresh-files").onclick = refreshFiles;
  $("#btn-refresh-streams").onclick = refreshStreams;
  $("#btn-open-path").onclick = () => {
    const p = $("#custom-path").value.trim();
    if (p) openVideo(p);
  };
  $("#btn-open-rtsp").onclick = () => {
    const url = $("#rtsp-url").value.trim();
    if (url) openVideo(url);
  };
  $("#rtsp-url").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") {
      const url = $("#rtsp-url").value.trim();
      if (url) openVideo(url);
    }
  });
  $("#btn-open-system-file").onclick = () => $("#video-file-picker").click();
  $("#video-file-picker").addEventListener("change", async (ev) => {
    const file = ev.target.files && ev.target.files[0];
    await openVideoFromSystemPicker(file);
    ev.target.value = "";
  });
  $("#btn-refresh-events").onclick = refreshEvents;
  $("#btn-clear-events").onclick = async () => {
    if (!confirm("Очистить журнал?")) return;
    await api("DELETE", "/api/events");
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
  renderTargetSelect();
  syncGridLayout();
  await loadSettings();
  await refreshStreams();
  await refreshFiles();
  await refreshEvents();
  switchPage("monitor");
  connectWs();
  state.streamsTimer = setInterval(refreshStreams, 4000);
}

main().catch(console.error);
