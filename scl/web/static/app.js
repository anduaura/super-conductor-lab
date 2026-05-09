"use strict";

const $ = (id) => document.getElementById(id);

// Color rotation for comparing N runs on the same charts.
const COMPARE_PALETTE = [
  "#58a6ff", "#ffa657", "#3fb950", "#f85149",
  "#a371f7", "#ff7b72", "#79c0ff", "#d2a8ff",
];
const COLORS = { active: "#58a6ff", baseline: "#ffa657" };

const AUTH_KEY = "scl-auth-token";

function getToken() {
  return localStorage.getItem(AUTH_KEY) || "";
}

function setToken(t) {
  if (t) localStorage.setItem(AUTH_KEY, t);
  else localStorage.removeItem(AUTH_KEY);
}

function authHeaders() {
  const t = getToken();
  return t ? { Authorization: `Bearer ${t}` } : {};
}

function streamUrl(path) {
  // EventSource cannot send headers; pass token as query param when set.
  const t = getToken();
  if (!t) return path;
  const sep = path.includes("?") ? "&" : "?";
  return `${path}${sep}token=${encodeURIComponent(t)}`;
}

async function authedFetch(url, options = {}) {
  const merged = { ...options, headers: { ...(options.headers || {}), ...authHeaders() } };
  const r = await fetch(url, merged);
  if (r.status === 401) {
    promptForToken();
  }
  return r;
}

async function promptForToken() {
  const t = window.prompt(
    "This server requires an auth token. Paste it here:",
    getToken(),
  );
  if (t != null) {
    setToken(t.trim());
    location.reload();
  }
}

async function ensureAuth() {
  // Probe healthz to learn whether auth is required, then validate token.
  const health = await fetch("/healthz").then(r => r.json()).catch(() => null);
  if (!health || !health.auth_required) return;
  const probe = await fetch("/api/auth", { headers: authHeaders() });
  if (probe.status === 401) {
    await promptForToken();
  }
}

let bestChart, scatterChart;
let activeStream = null;
let baselineStream = null;
let activeRunId = null;
let baselineRunId = null;

function initCharts() {
  const commonOpts = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    scales: {
      x: { ticks: { color: "#8b949e" }, grid: { color: "#242c36" }, title: { display: true, text: "round", color: "#8b949e" } },
      y: { ticks: { color: "#8b949e" }, grid: { color: "#242c36" }, title: { display: true, text: "Tc (K)", color: "#8b949e" } },
    },
    plugins: {
      legend: { labels: { color: "#e6edf3" } },
    },
  };

  bestChart = new Chart($("best-chart"), {
    type: "line",
    data: { datasets: [] },
    options: { ...commonOpts, parsing: { xAxisKey: "x", yAxisKey: "y" } },
  });

  scatterChart = new Chart($("scatter-chart"), {
    type: "scatter",
    data: { datasets: [] },
    options: {
      ...commonOpts,
      scales: {
        x: { ticks: { color: "#8b949e" }, grid: { color: "#242c36" }, title: { display: true, text: "predicted Tc (K)", color: "#8b949e" } },
        y: { ticks: { color: "#8b949e" }, grid: { color: "#242c36" }, title: { display: true, text: "measured Tc (K)", color: "#8b949e" } },
      },
    },
  });
}

function ensureSeries(label, color) {
  let bestIdx = bestChart.data.datasets.findIndex(d => d.label === `best so far (${label})`);
  if (bestIdx < 0) {
    bestChart.data.datasets.push({
      label: `best so far (${label})`,
      data: [], borderColor: color, backgroundColor: color, tension: 0.1, pointRadius: 2,
    });
    bestIdx = bestChart.data.datasets.length - 1;
  }
  let scatterIdx = scatterChart.data.datasets.findIndex(d => d.label === `predicted vs measured (${label})`);
  if (scatterIdx < 0) {
    scatterChart.data.datasets.push({
      label: `predicted vs measured (${label})`,
      data: [], borderColor: color, backgroundColor: color, pointRadius: 3,
    });
    scatterIdx = scatterChart.data.datasets.length - 1;
  }
  return { bestIdx, scatterIdx };
}

function clearCharts() {
  bestChart.data.datasets = [];
  scatterChart.data.datasets = [];
  bestChart.update("none");
  scatterChart.update("none");
  $("round-log").innerHTML = "";
}

function setStatus(text) {
  $("status-line").textContent = text;
}

function appendLog(prefix, ev, color) {
  const li = document.createElement("li");
  li.className = ev.success ? "ok" : "fail";
  const tag = ev.success ? "OK" : "FAIL";
  const tc = ev.measured_tc_k != null ? ev.measured_tc_k.toFixed(1) + "K" : "  --   ";
  const pred = ev.predicted_mean != null ?
    `${ev.predicted_mean.toFixed(1)}±${(ev.predicted_std ?? 0).toFixed(1)}` : " n/a ";
  li.style.color = color;
  li.textContent = `[${prefix}] r${String(ev.round).padStart(3,"0")} ${tag} pred=${pred} measured=${tc} best=${(ev.best_so_far_k ?? 0).toFixed(1)}K  ${ev.realized.formula} @ ${ev.realized.pressure_gpa.toFixed(0)}GPa  (${ev.note})`;
  $("round-log").prepend(li);
}

function pushRound(label, color, ev) {
  const { bestIdx, scatterIdx } = ensureSeries(label, color);
  bestChart.data.datasets[bestIdx].data.push({ x: ev.round, y: ev.best_so_far_k ?? 0 });
  if (ev.success && ev.predicted_mean != null && ev.measured_tc_k != null) {
    scatterChart.data.datasets[scatterIdx].data.push({ x: ev.predicted_mean, y: ev.measured_tc_k });
  }
  bestChart.update("none");
  scatterChart.update("none");
  appendLog(label, ev, color);
}

function streamRun(runId, label, color) {
  const es = new EventSource(streamUrl(`/api/runs/${runId}/stream`));
  let lastBest = 0;
  ensureSeries(label, color);
  es.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    if (ev.type === "round") {
      pushRound(label, color, ev);
      lastBest = ev.best_so_far_k ?? lastBest;
      setStatus(`[${label}] round ${ev.round}, best ${lastBest.toFixed(1)}K`);
    } else if (ev.type === "summary") {
      setStatus(`[${label}] done — best ${(ev.best_tc_k ?? 0).toFixed(1)}K (${ev.successful_rounds}/${ev.rounds} successful)`);
    } else if (ev.type === "error") {
      setStatus(`[${label}] error: ${ev.message?.split("\n")[0] ?? ev.message}`);
    } else if (ev.type === "closed") {
      es.close();
      refreshHistory();
    }
  };
  es.onerror = () => { es.close(); };
  return es;
}

async function startRun(formData) {
  const body = {};
  for (const [k, v] of formData.entries()) {
    if (k === "compare_baseline" || k === "use_agent") body[k] = true;
    else if (v === "" || v == null) continue;
    else if (k === "kappa" || k === "manifold_weight" || k === "target_tc_k") body[k] = parseFloat(v);
    else body[k] = parseInt(v, 10);
  }
  if (body.compare_baseline === undefined) body.compare_baseline = false;
  if (body.use_agent === undefined) body.use_agent = false;

  const r = await authedFetch("/api/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    setStatus(`error: ${r.status} ${await r.text()}`);
    return;
  }
  const { run_id, baseline_id } = await r.json();

  if (activeStream) activeStream.close();
  if (baselineStream) baselineStream.close();
  clearCharts();

  activeRunId = run_id;
  baselineRunId = baseline_id;
  $("run-id-tag").textContent = `${run_id}` + (baseline_id ? ` ⊕ ${baseline_id}` : "");
  activeStream = streamRun(run_id, "active", COLORS.active);
  if (baseline_id) baselineStream = streamRun(baseline_id, "baseline", COLORS.baseline);
}

async function refreshHistory() {
  const r = await authedFetch("/api/runs");
  if (!r.ok) return;
  const items = await r.json();
  const tbody = $("history-table").tBodies[0];
  tbody.innerHTML = "";
  for (const it of items) {
    const tr = document.createElement("tr");
    const when = it.created_at ? new Date(it.created_at * 1000).toLocaleString() : "—";
    const best = it.best_tc_k != null ? it.best_tc_k.toFixed(1) + "K" : "—";
    const rounds = it.rounds != null ? `${it.successful_rounds}/${it.rounds}` : "—";
    const idLabel = `${it.id}${it.paired_baseline_id ? ` ⊕ ${it.paired_baseline_id}` : ""}`;
    tr.innerHTML =
      `<td><input type="checkbox" class="run-select" data-id="${it.id}" data-baseline="${it.paired_baseline_id || ""}" /></td>` +
      `<td>${idLabel}</td>` +
      `<td>${when}</td>` +
      `<td>${it.status}</td>` +
      `<td>${rounds}</td>` +
      `<td>${best}</td>` +
      `<td><button data-id="${it.id}" data-baseline="${it.paired_baseline_id || ""}" class="view-btn">view</button></td>` +
      `<td>` +
        `<a class="export-link" href="${streamUrl(`/api/runs/${it.id}/export.csv`)}">CSV</a> · ` +
        `<a class="export-link" href="${streamUrl(`/api/runs/${it.id}/export.json`)}">JSON</a>` +
      `</td>`;
    tbody.appendChild(tr);
  }
  tbody.querySelectorAll("button.view-btn").forEach(b => {
    b.addEventListener("click", () => loadHistorical([
      { id: b.dataset.id, label: b.dataset.id, color: COLORS.active },
      ...(b.dataset.baseline ? [{ id: b.dataset.baseline, label: b.dataset.baseline, color: COLORS.baseline }] : []),
    ]));
  });
}

async function loadHistorical(specs) {
  const fetchRun = async (rid) => (await authedFetch(`/api/runs/${rid}`)).json();
  if (activeStream) activeStream.close();
  if (baselineStream) baselineStream.close();
  clearCharts();

  const tag = specs.map(s => s.id).join(" ⊕ ") + " (replay)";
  $("run-id-tag").textContent = tag;
  setStatus(`replaying ${specs.length} run${specs.length > 1 ? "s" : ""}`);

  const summaries = [];
  for (const spec of specs) {
    let data;
    try {
      data = await fetchRun(spec.id);
    } catch (e) {
      continue;
    }
    for (const ev of data.events) {
      if (ev.type === "round") pushRound(spec.label, spec.color, ev);
    }
    if (data.summary) {
      summaries.push(`${spec.label}=${(data.summary.best_tc_k ?? 0).toFixed(1)}K`);
    }
  }
  setStatus(summaries.length ? `replay: ${summaries.join(", ")}` : `replay complete`);
}

async function compareSelected() {
  const checked = Array.from(document.querySelectorAll(".run-select:checked"));
  if (checked.length < 2) {
    setStatus("pick at least two runs to compare");
    return;
  }
  const specs = [];
  let i = 0;
  for (const cb of checked) {
    const color = COMPARE_PALETTE[i % COMPARE_PALETTE.length];
    specs.push({ id: cb.dataset.id, label: cb.dataset.id, color });
    i++;
    if (cb.dataset.baseline) {
      specs.push({
        id: cb.dataset.baseline,
        label: `${cb.dataset.baseline} (baseline)`,
        color: COMPARE_PALETTE[i % COMPARE_PALETTE.length],
      });
      i++;
    }
  }
  await loadHistorical(specs);
}

document.addEventListener("DOMContentLoaded", async () => {
  initCharts();
  await ensureAuth();
  $("run-form").addEventListener("submit", (e) => {
    e.preventDefault();
    startRun(new FormData(e.target));
  });
  $("refresh-history").addEventListener("click", refreshHistory);
  $("compare-selected").addEventListener("click", compareSelected);
  refreshHistory();
});
