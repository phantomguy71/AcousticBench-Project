const S = { view: "overview", overview: null, datasets: [], models: [], runs: [], pick: null, detail: null, comp: null, board: [], busy: false };
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
    if (["runs", "compare", "overview", "board"].includes(view)) S.runs = await api("/api/runs");
    if (view === "board") S.board = await api("/api/leaderboard");
    if (view === "runs" && S.pick) S.detail = await api(`/api/runs/${S.pick}`);
    draw();
  } catch (e) { say(e.message, true); }
}
function draw() {
  document.querySelectorAll("nav [data-view]").forEach((b) => b.setAttribute("aria-current", b.dataset.view === S.view ? "page" : "false"));
  const V = { overview: vOverview, datasets: vDatasets, evaluate: vEvaluate, runs: vRuns, compare: vCompare, board: vBoard };
  main.innerHTML = (V[S.view] || vOverview)();
}
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const opt = (v, t) => `<option value="${esc(v)}">${esc(t)}</option>`;
function vOverview() {
  const o = S.overview || {};
  return `<div class="grid">
    <div class="card"><p class="muted">Datasets</p><p class="stat">${o.datasets ?? 0}</p></div>
    <div class="card"><p class="muted">Models</p><p class="stat">${o.models ?? 0}</p></div>
    <div class="card"><p class="muted">Runs</p><p class="stat">${o.runs ?? 0}</p></div>
    <div class="card"><p class="muted">Tasks</p><p>${esc(o.tasks || "")}</p></div></div>
    <div class="panel"><h2>Workflow</h2>
    <p>Register a simulated set and a real set, score one model on each, then compare. Every run keeps config, predictions, and provenance.</p>
    <p class="muted">${esc(o.gap || "")}</p>
    <div><button class="primary" data-action="nav-evaluate">Start an evaluation</button>
    <button data-action="gen">Build demo sets</button>
    <button data-action="rooms">Build synth rooms</button></div></div>
    <div class="panel"><h2>Regressions</h2><pre>${esc(JSON.stringify(o.regressions || [], null, 2))}</pre></div>`;
}
function dsTable(rows) {
  if (!rows.length) return `<p class="muted">None yet. Build the demo sets first.</p>`;
  return `<table><tr><th>ID</th><th>Domain</th><th>Items</th><th>Name</th></tr>${rows.map((d) => `<tr><td>${esc(d.id)}</td><td>${esc(d.domain)}</td><td>${esc(d.inventory?.n_items ?? "?")}</td><td>${esc(d.name)}</td></tr>`).join("")}</table>`;
}
function vDatasets() {
  return `<div class="split"><div class="panel"><h2>Datasets</h2>${dsTable(S.datasets)}</div>
  <form class="panel form-grid" data-form="dataset"><h2>Add dataset</h2>
  <label>Name <input name="name" required></label>
  <label>Domain <select name="domain"><option value="simulated">simulated</option><option value="real">real</option></select></label>
  <label>Layout <select name="layout"><option value="class_folders">class folders</option><option value="csv">manifest.csv</option></select></label>
  <label>Folder <input name="root_path" required placeholder="C:\\recordings\\room-a"></label>
  <label>Description <textarea name="description"></textarea></label>
  <button class="primary" type="submit">Register</button></form></div>`;
}
function modelOpts() { return S.models.map((m) => opt(m.id, `${m.id} — ${m.name}`)).join(""); }
function dsOpts(domain) { return S.datasets.filter((d) => !domain || d.domain === domain).map((d) => opt(d.id, `${d.id} (${d.domain})`)).join(""); }
function vEvaluate() {
  return `<div class="split"><form class="panel form-grid" data-form="evaluate"><h2>Run evaluation</h2>
  <label>Model <select name="model_id">${modelOpts()}</select></label>
  <label>Mode <select name="mode"><option value="paired">Paired sim + real</option><option value="single">Single dataset</option></select></label>
  <label>Dataset <select name="dataset_id">${dsOpts()}</select></label>
  <label>Simulated <select name="sim_dataset_id">${dsOpts("simulated")}</select></label>
  <label>Real <select name="real_dataset_id">${dsOpts("real")}</select></label>
  <label>Rate <input name="sample_rate" type="number" value="16000"></label>
  <label>Max seconds <input name="max_seconds" type="number" value="30" step="0.1"></label>
  <label>Notes <textarea name="notes"></textarea></label>
  <button class="primary" type="submit" ${S.busy ? "disabled" : ""}>Start</button></form>
  <div class="panel"><h2>Models</h2><table><tr><th>ID</th><th>Name</th></tr>${S.models.map((m) => `<tr><td>${esc(m.id)}</td><td>${esc(m.name)}</td></tr>`).join("")}</table></div></div>`;
}
function vRuns() {
  const rows = S.runs.map((r) => `<tr><td><a href="#" data-action="pick" data-id="${esc(r.id)}">${esc(r.id)}</a></td><td>${esc(r.status)}</td><td>${esc(r.model_id)}</td><td>${esc(r.dataset_id)}</td></tr>`).join("");
  return `<div class="panel"><h2>Runs</h2><table><tr><th>ID</th><th>Status</th><th>Model</th><th>Dataset</th></tr>${rows || `<tr><td colspan=4>No runs</td></tr>`}</table>
  ${S.detail ? `<h3>${esc(S.detail.id)}</h3><pre>${esc(JSON.stringify(S.detail.summary || {}, null, 2))}</pre>` : ""}</div>`;
}
function vCompare() {
  return `<div class="panel"><h2>Compare</h2>
  <form class="form-grid" data-form="compare"><label>Sim run <select name="sim">${S.runs.map((r) => opt(r.id, r.id)).join("")}</select></label>
  <label>Real run <select name="real">${S.runs.map((r) => opt(r.id, r.id)).join("")}</select></label>
  <button class="primary" type="submit">Compare</button></form>
  ${S.comp ? `<pre>${esc(JSON.stringify(S.comp, null, 2))}</pre>` : `<p class="muted">Pick two completed runs.</p>`}</div>`;
}
function vBoard() {
  return `<div class="panel"><h2>Leaderboard (best per task × dataset)</h2><pre>${esc(JSON.stringify(S.board, null, 2))}</pre></div>`;
}
async function handle(el) {
  const a = el.dataset.action;
  if (a === "nav-evaluate") { go("evaluate"); return; }
  if (a === "gen") { say("Building demo sets…"); await api("/api/examples/generate", { method: "POST" }).catch(() => api("/api/datasets", { method: "GET" })); say("Done."); go(S.view); return; }
  if (a === "rooms") { say("Rooms need the CLI: echobench synth-rooms"); return; }
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
