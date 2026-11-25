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
function vOverview() {
  const o = S.overview || {};
  return `<div class="grid">
    <div class="card"><p class="muted">Datasets</p><p class="stat">${o.datasets ?? 0}</p></div>
    <div class="card"><p class="muted">Models</p><p class="stat">${o.models ?? 0}</p></div>
    <div class="card"><p class="muted">Runs</p><p class="stat">${o.runs ?? 0}</p></div>
    <div class="card"><p class="muted">Tasks</p><p>${esc(o.tasks || "")}</p></div></div>
    <div class="panel"><h2>How an evaluation flows</h2>
    <p>Pick one synthetic set and one measured set, run the same model on both, then open Compare. Each entry keeps its settings, clip outputs, and machine notes.</p>
    <p class="muted">${esc(o.gap || "")}</p>
    <div><button class="primary" data-action="nav-evaluate">Start a new evaluation</button>
    <button data-action="gen">Make demo sets</button>
    <button data-action="rooms">Make synthetic rooms</button></div></div>
    <div class="panel"><h2>Recent slips</h2><pre>${esc(JSON.stringify(o.regressions || [], null, 2))}</pre></div>`;
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
  return `<div class="panel"><h2>Runs</h2><table><tr><th>ID</th><th>State</th><th>Model</th><th>Set</th></tr>${rows || `<tr><td colspan=4>Nothing yet</td></tr>`}</table>
  ${S.detail ? `<h3>${esc(S.detail.id)}</h3><pre>${esc(JSON.stringify(S.detail.summary || {}, null, 2))}</pre>` : ""}</div>`;
}
function vCompare() {
  return `<div class="panel"><h2>Compare</h2>
  <form class="form-grid" data-form="compare"><label>Synthetic run <select name="sim">${S.runs.map((r) => opt(r.id, r.id)).join("")}</select></label>
  <label>Measured run <select name="real">${S.runs.map((r) => opt(r.id, r.id)).join("")}</select></label>
  <button class="primary" type="submit">Compare</button></form>
  ${S.comp ? `<pre>${esc(JSON.stringify(S.comp, null, 2))}</pre>` : `<p class="muted">Choose two finished runs.</p>`}</div>`;
}
function vLeaderboard() {
  return `<div class="panel"><h2>Leaderboard (best per task \u00d7 set)</h2><pre>${esc(JSON.stringify(S.leaderboard, null, 2))}</pre></div>`;
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
