const S = { view: "overview", overview: null, datasets: [], models: [], runs: [], pick: null, detail: null, comp: null, leaderboard: [], busy: false };
const main = document.querySelector("#main");
const flash = document.querySelector("#flash");
const say = (m, bad) => { flash.textContent = m || ""; if (bad) flash.style.color = "#a00000"; };

document.body.addEventListener("click", (ev) => {
  const nav = ev.target.closest("nav [data-view]");
  if (nav) { go(nav.dataset.view); return; }
  const act = ev.target.closest("[data-action]");
  if (!act || act.disabled) return;
  handle(act).catch((e) => say(e.message, true));
});
document.body.addEventListener("submit", (ev) => {
  const f = ev.target;
  if (!(f instanceof HTMLFormElement) || !f.dataset.form) return;
  ev.preventDefault();
  send(f).catch((e) => say(e.message, true));
});
go("overview");

async function api(path, opts) {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || `Request failed (${r.status})`);
  return body;
}
async function go(view) {
  S.view = view; say("");
  try {
    if (view === "overview") S.overview = await api("/api/overview");
    if (["datasets", "evaluate", "compare"].includes(view)) S.datasets = await api("/api/datasets");
    if (["evaluate", "overview"].includes(view)) S.models = await api("/api/models");
    if (["runs", "compare", "overview", "leaderboard"].includes(view)) S.runs = await api("/api/runs");
    if (view === "leaderboard") S.leaderboard = await api("/api/leaderboard");
    if (view === "runs" && S.pick) S.detail = await api(`/api/runs/${S.pick}`);
    draw();
  } catch (e) { say(e.message, true); }
}
function draw() {
  document.querySelectorAll("nav [data-view]").forEach((b) => b.setAttribute("aria-current", b.dataset.view === S.view ? "page" : "false"));
  const V = { overview: vOverview, datasets: vDatasets, evaluate: vEvaluate, runs: vRuns, compare: vCompare, leaderboard: vLeaderboard };
  main.innerHTML = (V[S.view] || vOverview)();
}
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const opt = (v, t) => `<option value="${esc(v)}">${esc(t)}</option>`;
const fmt = (v) => (v === null || v === undefined || (typeof v === "number" && isNaN(v)) ? "\u2014" : (typeof v === "number" ? (Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(4)) : String(v)));
function metricRows(s) {
  const keys = ["accuracy","macro_f1","weighted_f1","macro_precision","macro_recall","event_f1","precision","recall","mae","rmse","median_ae","r2","ece","mean_latency_ms","p50_latency_ms","n_evaluated","n_errors"];
  return keys.filter((k) => s[k] !== undefined && s[k] !== null).map((k) => `<tr><td>${esc(k)}</td><td>${fmt(s[k])}</td></tr>`).join("");
}
function byCondTable(by) {
  if (!by || !Object.keys(by).length) return "";
  const rows = Object.entries(by).map(([g, r]) => `<tr><td>${esc(g)}</td><td>${esc(r.n ?? "")}</td><td>${fmt(r.mae)}</td><td>${fmt(r.rmse)}</td></tr>`).join("");
  return `<h3>By condition</h3><table><tr><th>Group</th><th>n</th><th>MAE</th><th>RMSE</th></tr>${rows}</table>`;
}
function perClassTable(pc) {
  if (!pc || !Object.keys(pc).length) return "";
  const rows = Object.entries(pc).map(([c, r]) => `<tr><td>${esc(c)}</td><td>${fmt(r.precision)}</td><td>${fmt(r.recall)}</td><td>${fmt(r.f1)}</td><td>${esc(r.support ?? "")}</td></tr>`).join("");
  return `<h3>Per class</h3><table><tr><th>Class</th><th>P</th><th>R</th><th>F1</th><th>n</th></tr>${rows}</table>`;
}
function warnList(w) {
  if (!w || !w.length) return "";
  return `<h3>Warnings</h3><ul>${w.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>`;
}
function renderRunDetail(d) {
  if (!d) return '<p class="muted">Select a run on the left.</p>';
  if (d.status !== "completed") return `<h3>${esc(d.id)}</h3><p><b>Status:</b> ${esc(d.status)}</p>` + (d.error ? `<p style="color:#ff7a7a">${esc(d.error)}</p><p class="muted">Tip: T60/distance models need RIR sets (synth-rooms), not tone/noise demo sets.</p>` : "");
  const s = d.summary || {};
  return `<h3>${esc(d.id)}</h3>
  <p><b>${esc(d.model_name || d.model_id || "")}</b> on <b>${esc(d.dataset_name || d.dataset_id || "")}</b> (${esc(d.dataset_domain || "")})</p>
  <table><tr><th>Metric</th><th>Value</th></tr>${metricRows(s)}</table>
  ${byCondTable(s.by_condition)}${perClassTable(s.per_class)}${warnList(s.warnings)}
  <p class="muted">Model: ${esc((s.model_info && s.model_info.id) || d.model_id || "")} \u00b7 task ${esc(s.task || "")} \u00b7 isolation ${esc(s.isolation || "")}</p>
  <p><a href="/api/runs/${esc(d.id)}/report.md">Download markdown report</a></p>
  <details><summary class="muted">Provenance + full JSON (debug)</summary><pre>${esc(JSON.stringify({summary: s, provenance: d.provenance || {}, config: d.config || {}}, null, 2).slice(0, 4000)}</pre></details>`;
}
function renderCompare(c) {
  if (!c) return '<p class="muted">Choose two runs on the left, then Compare.</p>';
  const mrows = (c.metrics || []).map((m) => `<tr><td>${esc(m.label)}</td><td>${fmt(m.simulated)}</td><td>${fmt(m.real)}</td><td>${fmt(m.gap)}</td></tr>`).join("");
  const prows = (c.per_class_f1 || []).map((r) => `<tr><td>${esc(r.label)}</td><td>${fmt(r.simulated_f1)}</td><td>${fmt(r.real_f1)}</td><td>${fmt(r.gap)}</td></tr>`).join("");
  const boot = ((c.significance || {}).bootstrap) || {};
  const warns = warnList(c.warnings);
  return `<p class="muted">${esc(c.gap_definition || "gap = real \u2212 simulated")} \u00b7 task ${esc(c.task || "")}</p>
  <table><tr><th>Metric</th><th>Sim</th><th>Real</th><th>Gap</th></tr>${mrows}</table>`
  + (prows ? `<h3>Per-class F1</h3><table><tr><th>Class</th><th>Sim</th><th>Real</th><th>Gap</th></tr>${prows}</table>` : "")
  + `<h3>Significance</h3><p>Bootstrap gap ${fmt(boot.gap)} [${fmt(boot.ci_low)}, ${fmt(boot.ci_high)}], ${esc(boot.rounds ?? 0)} resamples. Sign-test p: ${fmt(c.significance && c.significance.sign_p)}</p>` + warns;
}
function boardRows(b) {
  const head = (s) => s.macro_f1 !== undefined && s.macro_f1 !== null ? "F1 " + fmt(s.macro_f1) : s.event_f1 !== undefined && s.event_f1 !== null ? "F1 " + fmt(s.event_f1) : s.mae !== undefined && s.mae !== null ? "MAE " + fmt(s.mae) : "";
  return (b || []).map((r) => { const s = r.score || {}; return `<tr><td>${esc(r.task)}</td><td>${esc(r.dataset_id)}</td><td>${esc(r.run_id)}</td><td>${esc(r.model_id || "")}</td><td>${esc(head(s))}</td></tr>`; }).join("");
}
function vOverview() {
  const o = S.overview || {};
  return `<div class="split"><div><div class="grid">
    <div class="card"><p class="muted">Datasets</p><p class="stat">${o.datasets ?? 0}</p></div>
    <div class="card"><p class="muted">Models</p><p class="stat">${o.models ?? 0}</p></div>
    <div class="card"><p class="muted">Runs</p><p class="stat">${o.runs ?? 0}</p></div>
    <div class="card"><p class="muted">Tasks</p><p>${esc(o.tasks || "")}</p></div></div></div>
    <div class="panel" style="margin-top:0"><h2>How an evaluation flows</h2>
    <p>Pick one synthetic set and one measured set, run the same model on both, then open Compare. Each entry keeps its settings, clip outputs, and machine notes.</p>
    <p class="muted">${esc(o.gap || "")}</p>
    <div><button class="primary" data-action="nav-evaluate">Start a new evaluation</button>
    <button data-action="gen">Make demo sets</button>
    <button data-action="rooms">Make synthetic rooms</button></div></div>
    <div class="panel"><h2>Recent slips</h2><pre>${esc(JSON.stringify(o.regressions || [], null, 2))}</pre></div></div>`;
}
function dsTable(rows) {
  if (!rows.length) return `<p class="muted">Nothing here yet. Make the practice sets first.</p>`;
  return `<table><tr><th>ID</th><th>Kind</th><th>Clips</th><th>Title</th></tr>${rows.map((d) => `<tr><td>${esc(d.id)}</td><td>${esc(d.domain)}</td><td>${esc(d.inventory?.n_items ?? "?")}</td><td>${esc(d.name)}</td></tr>`).join("")}</table>`;
}
function vDatasets() {
  return `<div class="split"><div class="panel"><h2>Datasets</h2>${dsTable(S.datasets)}</div>
  <form class="panel form-grid" data-form="dataset"><h2>Add a set</h2>
  <label>Title <input name="name" required></label>
  <label>Kind <select name="domain"><option value="simulated">simulated</option><option value="real">real</option></select></label>
  <label>Shape <select name="layout"><option value="class_folders">class folders</option><option value="csv">manifest.csv</option></select></label>
  <label>Folder <input name="root_path" required placeholder="C:\\recordings\\room-a"></label>
  <label>Notes <textarea name="description"></textarea></label>
  <button class="primary" type="submit">Save set</button></form></div>`;
}
function modelOpts() { return S.models.map((m) => opt(m.id, `${m.id} — ${m.name}`)).join(""); }
function dsOpts(domain) { return S.datasets.filter((d) => !domain || d.domain === domain).map((d) => opt(d.id, `${d.id} (${d.domain})`)).join(""); }
function vEvaluate() {
  return `<div class="split"><form class="panel form-grid" data-form="evaluate"><h2>Evaluate</h2>
  <label>Model <select name="model_id">${modelOpts()}</select></label>
  <label>Style <select name="mode"><option value="paired">Paired sim + real</option><option value="single">One set only</option></select></label>
  <label>Set <select name="dataset_id">${dsOpts()}</select></label>
  <label>Synthetic <select name="sim_dataset_id">${dsOpts("simulated")}</select></label>
  <label>Measured <select name="real_dataset_id">${dsOpts("real")}</select></label>
  <label>Rate <input name="sample_rate" type="number" value="16000"></label>
  <label>Clip cap (s) <input name="max_seconds" type="number" value="30" step="0.1"></label>
  <label>Notes <textarea name="notes"></textarea></label>
  <button class="primary" type="submit" ${S.busy ? "disabled" : ""}>Launch</button></form>
  <div class="panel"><h2>Models</h2><table><tr><th>ID</th><th>Title</th></tr>${S.models.map((m) => `<tr><td>${esc(m.id)}</td><td>${esc(m.name)}</td></tr>`).join("")}</table></div></div>`;
}
function vRuns() {
  const rows = S.runs.map((r) => `<tr><td><a href="#" data-action="pick" data-id="${esc(r.id)}">${esc(r.id)}</a></td><td>${esc(r.status)}</td><td>${esc(r.model_id)}</td><td>${esc(r.dataset_id)}</td></tr>`).join("");
  return `<div class="split"><div class="panel" style="margin-top:0"><h2>Runs</h2><table><tr><th>ID</th><th>State</th><th>Model</th><th>Set</th></tr>${rows || `<tr><td colspan=4>Nothing yet</td></tr>`}</table><p class="muted">Click a run to inspect it on the side.</p></div>
  <div class="panel" style="margin-top:0"><h2>Run detail</h2>
  ${renderRunDetail(S.detail)}</div></div>`;
}
function vCompare() {
  return `<div class="split"><div class="panel" style="margin-top:0"><h2>Compare</h2>
  <form class="form-grid" data-form="compare"><label>Synthetic run <select name="sim">${S.runs.map((r) => opt(r.id, r.id)).join("")}</select></label>
  <label>Measured run <select name="real">${S.runs.map((r) => opt(r.id, r.id)).join("")}</select></label>
  <button class="primary" type="submit">Compare</button></form>
  </div><div class="panel" style="margin-top:0"><h2>Result</h2>${renderCompare(S.comp)}</div></div>`;
}
function vLeaderboard() {
  return `<div class="split"><div class="panel" style="margin-top:0"><h2>Leaderboard (best per task \u00d7 set)</h2><table><tr><th>Task</th><th>Set</th><th>Run</th><th>Model</th><th>Score</th></tr>${boardRows(S.leaderboard)}</table></div><div class="panel" style="margin-top:0"><h2>How scoring works</h2><p>Higher F1 wins for room/classes/events. Lower MAE wins for t60/distance.</p><p class="muted">gap = real \u2212 simulated</p></div></div>`;
}
async function handle(el) {
  const a = el.dataset.action;
  if (a === "nav-evaluate") { go("evaluate"); return; }
  if (a === "gen") { say("Making demo sets…"); await api("/api/examples/generate", { method: "POST" }).catch(() => api("/api/datasets", { method: "GET" })); say("Done."); go(S.view); return; }
  if (a === "rooms") { say("Rooms need the shell: echobench synth-rooms"); return; }
  if (a === "pick") { S.pick = el.dataset.id; S.detail = await api(`/api/runs/${S.pick}`); draw(); return; }
}
async function send(f) {
  const data = Object.fromEntries(new FormData(f).entries());
  if (f.dataset.form === "dataset") { await api("/api/datasets", { method: "POST", body: JSON.stringify(data) }); say("Dataset registered."); }
  if (f.dataset.form === "evaluate") {
    S.busy = true; draw();
    const body = { ...data, sample_rate: Number(data.sample_rate), max_seconds: Number(data.max_seconds) };
    if (data.mode === "paired") await api("/api/evaluations/paired", { method: "POST", body: JSON.stringify(body) });
    else await api("/api/evaluations", { method: "POST", body: JSON.stringify(body) });
    say("Run queued; watch the Runs tab.");
  }
  if (f.dataset.form === "compare") { S.comp = await api(`/api/compare?sim=${encodeURIComponent(data.sim)}&real=${encodeURIComponent(data.real)}`); }
  S.busy = false; go(S.view);
}
