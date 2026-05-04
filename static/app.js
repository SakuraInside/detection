/* Front-end Integra-LOST.
   Сборка не нужна: чистый ES2020 в браузере. */

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const state = {
  loaded: false,
  playing: false,
  totalFrames: 0,
  fps: 30,
  seekDragging: false,
  // Храним активные тревоги в памяти UI: ключ = track_id.
  alarms: new Map(),
  ws: null,
  settings: null,
  metricsTimer: null,
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
  const root = $(rootSel);
  if (!root) return;
  const p = Math.max(0, Math.min(100, Number.isFinite(pct) ? pct : 0));
  root.style.setProperty("--pct", String(p));
  const valEl = root.querySelector(".speedo-value");
  const subEl = root.querySelector(".speedo-sub");
  if (valEl) valEl.textContent = valueText;
  if (subEl) subEl.textContent = subText;
}

let metricsTimer = null;

async function refreshMetrics() {
  try {
    const data = await api("GET", "/api/metrics");
    const sys = data.system || {};
    const proc = data.process || {};
    const gpu = sys.gpu || {};

    const cpuPct = sys.cpu_percent;
    updateSpeedo(
      "#speedo-cpu",
      cpuPct ?? 0,
      cpuPct != null ? `${cpuPct.toFixed(0)}%` : "—",
      proc.cpu_percent != null ? `процесс ${proc.cpu_percent.toFixed(0)}%` : ""
    );

    const ramPct = sys.ram_percent;
    const ramSub =
      sys.ram_used_bytes != null && sys.ram_total_bytes != null
        ? `${fmtBytes(sys.ram_used_bytes)} / ${fmtBytes(sys.ram_total_bytes)}`
        : "";
    updateSpeedo("#speedo-ram", ramPct ?? 0, ramPct != null ? `${ramPct.toFixed(0)}%` : "—", ramSub);

    const gn = $("#gpu-name");
    if (gpu.available && gpu.util_percent != null) {
      const memSub =
        gpu.memory_used_bytes != null && gpu.memory_total_bytes != null
          ? `${fmtBytes(gpu.memory_used_bytes)} / ${fmtBytes(gpu.memory_total_bytes)}`
          : "";
      updateSpeedo("#speedo-gpu", gpu.util_percent, `${gpu.util_percent.toFixed(0)}%`, memSub);
      if (gn) gn.textContent = gpu.name || "GPU";
    } else {
      updateSpeedo("#speedo-gpu", 0, "—", "нет данных");
      if (gn) gn.textContent = "GPU не обнаружена (nvidia-smi / драйвер)";
    }
  } catch (e) {
    console.warn("metrics fail", e);
  }
}

function switchOperativeTab(tab) {
  const evBtn = $("#tab-events");
  const sysBtn = $("#tab-system");
  const panelEv = $("#panel-events");
  const panelSys = $("#panel-system");
  if (!evBtn || !sysBtn || !panelEv || !panelSys) return;
  if (tab === "events") {
    evBtn.classList.add("tab-btn-active");
    sysBtn.classList.remove("tab-btn-active");
    panelEv.classList.remove("hidden");
    panelSys.classList.add("hidden");
    if (metricsTimer != null) {
      clearInterval(metricsTimer);
      metricsTimer = null;
    }
  } else {
    sysBtn.classList.add("tab-btn-active");
    evBtn.classList.remove("tab-btn-active");
    panelSys.classList.remove("hidden");
    panelEv.classList.add("hidden");
    refreshMetrics();
    if (metricsTimer == null) {
      metricsTimer = setInterval(refreshMetrics, 2500);
    }
  }
}

async function api(method, path, body) {
  const opts = { method, headers: { "Content-Type": "application/json" } };
  if (body !== undefined) opts.body = JSON.stringify(body);
  const r = await fetch(path, opts);
  if (!r.ok) {
    const text = await r.text();
    throw new Error(`${r.status}: ${text}`);
  }
  return r.json();
}

// ----------------------------------------------------------- опрос статуса

async function refreshInfo() {
  try {
    const info = await api("GET", "/api/info");
    applyInfo(info);
  } catch (e) {
    console.warn("info fail", e);
  }
}

async function refreshMetrics() {
  try {
    const data = await api("GET", "/api/metrics");
    applySystemMetrics(data.system || {});
  } catch (e) {
    console.warn("metrics fail", e);
  }
}

function fmtBytes(v) {
  const n = Number(v || 0);
  if (!isFinite(n) || n <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let x = n;
  let i = 0;
  while (x >= 1024 && i < units.length - 1) {
    x /= 1024;
    i += 1;
  }
  return `${x.toFixed(i <= 1 ? 0 : 1)} ${units[i]}`;
}

function updateSpeedo(id, pct, usedText, totalText) {
  const root = document.getElementById(id);
  if (!root) return;
  const val = Math.max(0, Math.min(100, Number(pct || 0)));
  root.style.setProperty("--pct", String(val));
  const valueEl = root.querySelector(".speedo-value");
  const subEl = root.querySelector(".speedo-sub");
  if (valueEl) valueEl.textContent = `${val.toFixed(0)}%`;
  if (subEl) subEl.textContent = `${usedText} / ${totalText}`;
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
  $("#dropped-info").textContent = `пропущено ${s.dropped_decode || 0}`;

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
    const data = await api("GET", "/api/events?limit=200");
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
      ? `<a href="/${ev.snapshot_path}" target="_blank" rel="noopener noreferrer" title="Открыть снимок">
           <img src="/${ev.snapshot_path}" alt="снимок события"/>
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
  const data = await api("GET", "/api/files");
  const root = $("#files-list");
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
    await api("POST", "/api/open", { path });
    await refreshInfo();
    await refreshEvents();
    // Принудительно переподключаем MJPEG-поток через смену query-параметра.
    const v = $("#video");
    v.src = "/video_feed?ts=" + Date.now();
  } catch (e) { alert("Не удалось открыть: " + e.message); }
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
    if (msg.type === "status") applyInfo(msg.info);
    else if (msg.type === "hello") applyInfo(msg.info);
    else if (msg.type === "event") onEventMessage(msg);
  };
}

function onEventMessage(ev) {
  refreshEvents();
  if (ev.type === "abandoned" || ev.type === "disappeared") {
    if (state.settings?.ui?.alarm_sound) {
      try { $("#alarm-sound").currentTime = 0; $("#alarm-sound").play(); } catch {}
    }
  }
}

// ----------------------------------------------------------- привязка обработчиков UI

function wireControls() {
  $("#btn-play").onclick = () => api("POST", "/api/play").then(refreshInfo);
  $("#btn-pause").onclick = () => api("POST", "/api/pause").then(refreshInfo);

  const seek = $("#seek");
  seek.addEventListener("mousedown", () => state.seekDragging = true);
  seek.addEventListener("touchstart", () => state.seekDragging = true);
  const release = () => {
    if (!state.seekDragging) return;
    state.seekDragging = false;
    // Отправляем seek один раз при отпускании ползунка.
    api("POST", "/api/seek", { frame: Number(seek.value) }).then(refreshInfo);
  };
  seek.addEventListener("mouseup", release);
  seek.addEventListener("touchend", release);
  seek.addEventListener("change", () => {
    if (!state.seekDragging) {
      api("POST", "/api/seek", { frame: Number(seek.value) }).then(refreshInfo);
    }
  });

  $("#btn-refresh-files").onclick = refreshFiles;
  $("#btn-open-path").onclick = () => {
    const p = $("#custom-path").value.trim();
    if (p) openVideo(p);
  };
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

  $("#tab-events").onclick = () => switchOperativeTab("events");
  $("#tab-system").onclick = () => switchOperativeTab("system");
}

async function main() {
  wireControls();
  await loadSettings();
  await refreshFiles();
  await refreshEvents();
  await refreshInfo();
  await refreshMetrics();
  connectWs();
  state.metricsTimer = setInterval(refreshMetrics, 1500);
}

main().catch(console.error);
