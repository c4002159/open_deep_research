"""Single-file admin dashboard. The page itself carries no data; it asks for an admin bearer token
(kept in sessionStorage only) and renders /v1/admin/summary."""

DASHBOARD_HTML = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Usage dashboard</title>
<style>
  :root { --bg:#fafaf9; --fg:#1c1917; --mut:#78716c; --card:#fff; --line:#e7e5e4; --acc:#2563eb; --bad:#dc2626; }
  @media (prefers-color-scheme: dark) { :root { --bg:#1c1917; --fg:#f5f5f4; --mut:#a8a29e; --card:#292524; --line:#44403c; --acc:#60a5fa; --bad:#f87171; } }
  body { margin:0; background:var(--bg); color:var(--fg); font:14px/1.5 system-ui,sans-serif; }
  main { max-width:1000px; margin:0 auto; padding:24px 16px; }
  h1 { font-size:20px; margin:0 0 4px } .sub { color:var(--mut); margin-bottom:20px }
  .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin-bottom:20px }
  .card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px 14px }
  .card b { display:block; font-size:22px } .card span { color:var(--mut); font-size:12px }
  table { width:100%; border-collapse:collapse; background:var(--card); border:1px solid var(--line); border-radius:8px; overflow:hidden; margin-bottom:20px }
  th,td { text-align:right; padding:8px 12px; border-bottom:1px solid var(--line) } th:first-child,td:first-child { text-align:left }
  th { color:var(--mut); font-weight:500; font-size:12px } .bar { height:8px; background:var(--acc); border-radius:4px; min-width:2px }
  .bad { color:var(--bad) } #msg { color:var(--bad); margin:12px 0 }
  h2 { font-size:15px; margin:24px 0 8px } button { font:inherit; padding:4px 10px }
</style></head><body><main>
<h1>Usage dashboard</h1><div class="sub" id="sub">loading…</div><div id="msg"></div>
<div class="cards" id="cards"></div>
<h2>Per user (today, UTC)</h2><table id="users"></table>
<h2>Tools</h2><table id="tools"></table>
<h2>Task latency (submit → finished)</h2><table id="lat"></table>
<h2>Counters</h2><table id="ctr"></table>
<button onclick="sessionStorage.removeItem('t');load()">change token</button> <button onclick="load()">refresh</button>
</main><script>
const fmt = n => n == null ? "–" : n.toLocaleString();
const money = v => v == null ? "–" : "$" + v.toFixed(4);   // null = prices not configured
async function load() {
  let t = sessionStorage.getItem("t");
  if (!t) { t = prompt("Admin bearer token"); if (!t) return; sessionStorage.setItem("t", t); }
  const r = await fetch("/v1/admin/summary", { headers: { Authorization: "Bearer " + t } });
  if (!r.ok) { document.getElementById("msg").textContent = "HTTP " + r.status + " - " + (await r.text()).slice(0, 120); sessionStorage.removeItem("t"); return; }
  document.getElementById("msg").textContent = "";
  const d = await r.json(), g = d.totals;
  document.getElementById("sub").textContent = d.day + " · queue depth " + d.queue_depth +
    (g.est_cost_usd == null ? " · cost shows – until ODR_PRICE_IN_PER_M / ODR_PRICE_OUT_PER_M are set" : "");
  const cards = [["tokens in", fmt(g.tokens_in)], ["tokens out", fmt(g.tokens_out)], ["est. cost", money(g.est_cost_usd)],
                 ["LLM calls", fmt(g.llm_calls)], ["tool calls", fmt(g.tool_calls)], ["tool errors", fmt(g.tool_errors)]];
  document.getElementById("cards").innerHTML = cards.map(c => `<div class="card"><b>${c[1]}</b><span>${c[0]}</span></div>`).join("");
  const max = Math.max(1, ...d.users.map(u => u.tokens_in + u.tokens_out));
  document.getElementById("users").innerHTML = "<tr><th>user</th><th>runs</th><th>tokens</th><th>est. cost</th><th style='width:30%'></th></tr>" +
    d.users.map(u => `<tr><td>${u.user}</td><td>${fmt(u.runs)}</td><td>${fmt(u.tokens_in + u.tokens_out)}</td><td>${money(u.est_cost_usd)}</td>` +
      `<td><div class="bar" style="width:${100 * (u.tokens_in + u.tokens_out) / max}%"></div></td></tr>`).join("");
  document.getElementById("tools").innerHTML = "<tr><th>tool</th><th>calls</th><th>errors</th><th>error rate</th></tr>" +
    d.tools.map(x => `<tr><td>${x.tool}</td><td>${fmt(x.calls)}</td><td>${fmt(x.errors)}</td><td class="${x.error_rate > 0.2 ? 'bad' : ''}">${(100 * x.error_rate).toFixed(1)}%</td></tr>`).join("");
  document.getElementById("lat").innerHTML = "<tr><th>≤ seconds</th><th>tasks</th></tr>" +
    Object.entries(d.latency).map(([k, v]) => `<tr><td>${k.replace("le_", "")}</td><td>${fmt(v)}</td></tr>`).join("");
  document.getElementById("ctr").innerHTML = Object.entries(d.counters).map(([k, v]) => `<tr><td>${k}</td><td>${fmt(v)}</td></tr>`).join("");
}
load();
</script></body></html>"""
