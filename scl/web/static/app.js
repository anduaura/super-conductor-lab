"use strict";

const $ = (id) => document.getElementById(id);

const COLORS = {
  active: "#58a6ff",
  baseline: "#ffa657",
};

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
    data: { datasets: [
      { label: "best so far (active)", data: [], borderColor: COLORS.active, backgroundColor: COLORS.active, tension: 0.1 },
      { label: "best so far (baseline)", data: [], borderColor: COLORS.baseline, backgroundColor: COLORS.baseline, tension: 0.1, hidden: true },
    ] },
    options: { ...commonOpts, parsing: { xAxisKey: "x", yAxisKey: "y" } },
  });

  scatterChart = new Chart($("scatter-chart"), {
    type: "scatter",
    data: { datasets: [
      { label: "predicted vs measured (active)", data: [], borderColor: COLORS.active, backgroundColor: COLORS.active },
      { label: "predicted vs measured (baseline)", data: [], borderColor: COLORS.baseline, backgroundColor: COLORS.baseline, hidden: true },
    ] },
    options: {
      ...commonOpts,
      scales: {
        x: { ticks: { color: "#8b949e" }, grid: { color: "#242c36" }, title: { display: true, text: "predicted Tc (K)", color: "#8b949e" } },
        y: { ticks: { color: "#8b949e" }, grid: { color: "#242c36" }, title: { display: true, text: "measured Tc (K)", color: "#8b949e" } },
      },
    },
  });
}

function clearCharts() {
  bestChart.data.datasets.forEach(d => d.data = []);
  scatterChart.data.datasets.forEach(d => d.data = []);
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

function pushRound(which, datasetIndex, ev) {
  const bestDS = bestChart.data.datasets[datasetIndex];
  bestDS.data.push({ x: ev.round, y: ev.best_so_far_k ?? 0 });
  bestDS.hidden = false;
  if (ev.success && ev.predicted_mean != null && ev.measured_tc_k != null) {
    const scatterDS = scatterChart.data.datasets[datasetIndex];
    scatterDS.data.push({ x: ev.predicted_mean, y: ev.measured_tc_k });
    scatterDS.hidden = false;
  }
  bestChart.update("none");
  scatterChart.update("none");
  appendLog(which, ev, datasetIndex === 0 ? COLORS.active : COLORS.baseline);
}

function streamRun(runId, label, datasetIndex) {
  const es = new EventSource(`/api/runs/${runId}/stream`);
  let lastBest = 0;
  es.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    if (ev.type === "round") {
      pushRound(label, datasetIndex, ev);
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

  const r = await fetch("/api/runs", {
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
  activeStream = streamRun(run_id, "active", 0);
  if (baseline_id) baselineStream = streamRun(baseline_id, "baseline", 1);
}

async function refreshHistory() {
  const r = await fetch("/api/runs");
  if (!r.ok) return;
  const items = await r.json();
  const tbody = $("history-table").tBodies[0];
  tbody.innerHTML = "";
  for (const it of items) {
    const tr = document.createElement("tr");
    const when = it.created_at ? new Date(it.created_at * 1000).toLocaleString() : "—";
    const best = it.best_tc_k != null ? it.best_tc_k.toFixed(1) + "K" : "—";
    const rounds = it.rounds != null ? `${it.successful_rounds}/${it.rounds}` : "—";
    tr.innerHTML = `<td>${it.id}${it.paired_baseline_id ? ` ⊕ ${it.paired_baseline_id}` : ""}</td><td>${when}</td><td>${it.status}</td><td>${rounds}</td><td>${best}</td><td><button data-id="${it.id}" data-baseline="${it.paired_baseline_id || ""}">view</button></td>`;
    tbody.appendChild(tr);
  }
  tbody.querySelectorAll("button[data-id]").forEach(b => {
    b.addEventListener("click", () => loadHistorical(b.dataset.id, b.dataset.baseline || null));
  });
}

async function loadHistorical(id, baselineId) {
  const fetchRun = async (rid) => (await fetch(`/api/runs/${rid}`)).json();
  if (activeStream) activeStream.close();
  if (baselineStream) baselineStream.close();
  clearCharts();
  $("run-id-tag").textContent = `${id}` + (baselineId ? ` ⊕ ${baselineId}` : "") + " (replay)";
  setStatus(`replaying ${id}`);
  const main = await fetchRun(id);
  for (const ev of main.events) {
    if (ev.type === "round") pushRound("active", 0, ev);
  }
  if (main.summary) setStatus(`[active] best ${(main.summary.best_tc_k ?? 0).toFixed(1)}K (${main.summary.successful_rounds}/${main.summary.rounds})`);
  if (baselineId) {
    const bl = await fetchRun(baselineId);
    for (const ev of bl.events) {
      if (ev.type === "round") pushRound("baseline", 1, ev);
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  initCharts();
  $("run-form").addEventListener("submit", (e) => {
    e.preventDefault();
    startRun(new FormData(e.target));
  });
  $("refresh-history").addEventListener("click", refreshHistory);
  refreshHistory();
});
