"use strict";

const COLORS = { active: "#58a6ff", baseline: "#ffa657" };

const $ = (id) => document.getElementById(id);

const COMMON = {
  responsive: true,
  maintainAspectRatio: false,
  animation: false,
  scales: {
    x: { ticks: { color: "#8b949e" }, grid: { color: "#242c36" }, title: { display: true, color: "#8b949e" } },
    y: { ticks: { color: "#8b949e" }, grid: { color: "#242c36" }, title: { display: true, color: "#8b949e" } },
  },
  plugins: { legend: { labels: { color: "#e6edf3" } } },
};

function makeCharts() {
  const best = new Chart($("best-chart"), {
    type: "line",
    data: { datasets: [
      { label: "best so far (active)", data: [], borderColor: COLORS.active, backgroundColor: COLORS.active, tension: 0.1, pointRadius: 2 },
      { label: "best so far (baseline)", data: [], borderColor: COLORS.baseline, backgroundColor: COLORS.baseline, tension: 0.1, pointRadius: 2 },
    ] },
    options: {
      ...COMMON,
      scales: {
        x: { ...COMMON.scales.x, title: { ...COMMON.scales.x.title, text: "round" } },
        y: { ...COMMON.scales.y, title: { ...COMMON.scales.y.title, text: "best Tc (K)" } },
      },
    },
  });
  const scatter = new Chart($("scatter-chart"), {
    type: "scatter",
    data: { datasets: [
      { label: "predicted vs measured (active)", data: [], borderColor: COLORS.active, backgroundColor: COLORS.active, pointRadius: 3 },
      { label: "predicted vs measured (baseline)", data: [], borderColor: COLORS.baseline, backgroundColor: COLORS.baseline, pointRadius: 3 },
    ] },
    options: {
      ...COMMON,
      scales: {
        x: { ...COMMON.scales.x, title: { ...COMMON.scales.x.title, text: "predicted Tc (K)" } },
        y: { ...COMMON.scales.y, title: { ...COMMON.scales.y.title, text: "measured Tc (K)" } },
      },
    },
  });
  return { best, scatter };
}

async function loadJsonl(url) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`failed to load ${url}: ${resp.status}`);
  const text = await resp.text();
  const events = [];
  let meta = null;
  let summary = null;
  for (const line of text.split("\n")) {
    if (!line.trim()) continue;
    const obj = JSON.parse(line);
    if (obj.type === "meta") meta = obj;
    else if (obj.type === "summary") summary = obj;
    else events.push(obj);
  }
  return { meta, events, summary };
}

function renderRun({ events, summary }, datasetIdx, charts) {
  for (const ev of events) {
    if (ev.type !== "round") continue;
    const round = ev.round >= 0 ? ev.round : null;
    if (round !== null) {
      charts.best.data.datasets[datasetIdx].data.push({
        x: round, y: ev.best_so_far_k ?? 0,
      });
      if (ev.success && ev.predicted_mean != null && ev.measured_tc_k != null) {
        charts.scatter.data.datasets[datasetIdx].data.push({
          x: ev.predicted_mean, y: ev.measured_tc_k,
        });
      }
    }
  }
  return summary;
}

function appendLogLines(events, label, color) {
  const log = $("round-log");
  for (const ev of events) {
    if (ev.type !== "round") continue;
    const li = document.createElement("li");
    li.className = ev.success ? "ok" : "fail";
    li.style.color = color;
    const tag = ev.success ? "OK" : "FAIL";
    const tc = ev.measured_tc_k != null ? `${ev.measured_tc_k.toFixed(1)}K` : "  --   ";
    const pred = ev.predicted_mean != null
      ? `${ev.predicted_mean.toFixed(1)}±${(ev.predicted_std ?? 0).toFixed(1)}`
      : " n/a ";
    const r = ev.round >= 0 ? String(ev.round).padStart(3, "0") : "seed";
    li.textContent = `[${label}] r${r} ${tag} pred=${pred} measured=${tc} best=${(ev.best_so_far_k ?? 0).toFixed(1)}K  ${ev.realized.formula} @ ${ev.realized.pressure_gpa.toFixed(0)}GPa  (${ev.note})`;
    log.appendChild(li);
  }
}

(async () => {
  const charts = makeCharts();
  let activeData, baselineData;
  try {
    [activeData, baselineData] = await Promise.all([
      loadJsonl("active.jsonl"),
      loadJsonl("baseline.jsonl"),
    ]);
  } catch (e) {
    $("summary-line").textContent = `error loading run data: ${e.message}`;
    return;
  }

  const activeSummary = renderRun(activeData, 0, charts);
  const baselineSummary = renderRun(baselineData, 1, charts);
  charts.best.update("none");
  charts.scatter.update("none");

  appendLogLines(activeData.events, "active", COLORS.active);
  appendLogLines(baselineData.events, "baseline", COLORS.baseline);

  if (activeSummary && baselineSummary) {
    const ab = activeSummary.best_tc_k ?? 0;
    const bb = baselineSummary.best_tc_k ?? 0;
    const delta = ab - bb;
    const sign = delta >= 0 ? "+" : "";
    const aBest = activeSummary.best_candidate;
    const aBestStr = aBest ? `${aBest.formula} @ ${aBest.pressure_gpa.toFixed(0)}GPa` : "—";
    $("summary-line").innerHTML =
      `<span style="color:#58a6ff">active</span> ${ab.toFixed(1)}K ` +
      `<span style="color:#8b949e">(${activeSummary.successful_rounds}/${activeSummary.rounds} successful, best: ${aBestStr})</span>` +
      `&nbsp;&nbsp;vs&nbsp;&nbsp;` +
      `<span style="color:#ffa657">baseline</span> ${bb.toFixed(1)}K ` +
      `<span style="color:#8b949e">(${baselineSummary.successful_rounds}/${baselineSummary.rounds} successful)</span>` +
      `&nbsp;&nbsp;Δ ${sign}${delta.toFixed(1)}K`;
  }
})();
