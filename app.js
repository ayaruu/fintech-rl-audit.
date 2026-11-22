/* AuditRL frontend. Plain JavaScript, no build step. Talks to the FastAPI backend under /api. */
"use strict";

// =============================================================== helpers
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const money = n => Number(n).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const pct = x => (x * 100).toFixed(1) + "%";
const sleep = ms => new Promise(r => setTimeout(r, ms));
const signed = n => (n > 0 ? "+" : "") + n;

const COLORS = { rl_agent: "#2563eb", risk_threshold: "#0ea5e9", rule_based: "#7c8cf8", random: "#9aa7bd" };
const ACTION_TEXT = { approve: "Approve", flag: "Flag", request_docs: "Request docs", investigate: "Investigate" };
const OUTCOME_TEXT = {
  true_positive: ["Caught suspicious ✔", "ok"], true_negative: ["Correctly approved ✔", "ok"],
  false_positive: ["False alarm ✘", "bad"], false_negative: ["Missed suspicious ✘", "bad"],
};
const PAGES = { dashboard: "Dashboard", transactions: "Transactions", agent: "AI Audit Agent", reports: "Audit Reports", environment: "RL Environment" };

const S = { meta: null, session: null, animToken: 0, txRows: [], txPage: 1, demo: null, reportQuery: "" };

function setStatus(ok, text) {
  const p = $("#status-pill");
  p.textContent = text || (ok ? "Backend connected" : "Backend offline");
  p.className = "pill " + (ok ? "ok" : "bad");
}
let bannerTimer;
function showError(msg) {
  const b = $("#banner");
  b.textContent = msg; b.classList.remove("hidden");
  clearTimeout(bannerTimer); bannerTimer = setTimeout(() => b.classList.add("hidden"), 8000);
}
async function api(path, opts = {}) {
  let res;
  try {
    res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  } catch (e) {
    setStatus(false);
    throw new Error('Cannot reach the backend. Start it with "python run.py" and open http://localhost:8000');
  }
  if (!res.ok) {
    let detail = res.statusText;
    try { const j = await res.json(); detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch (_) { /* ignore */ }
    throw new Error(detail);
  }
  return res.json();
}
const post = (path, body) => api(path, { method: "POST", body: JSON.stringify(body || {}) });
function guard(fn) { // run an async handler and show errors in the red banner
  return async (...a) => { try { await fn(...a); } catch (e) { showError(e.message); } };
}
function fillSelect(sel, options, value) {
  sel.innerHTML = options.map(o => `<option value="${esc(o.value)}">${esc(o.label)}</option>`).join("");
  if (value !== undefined) sel.value = value;
}
function table(el, headers, rows, emptyText) {
  // headers: [{label, cls}] ; rows: array of arrays of HTML strings
  if (!rows.length) { el.innerHTML = `<tbody><tr><td class="empty">${esc(emptyText || "No data")}</td></tr></tbody>`; return; }
  el.innerHTML = "<thead><tr>" + headers.map(h => `<th class="${h.cls || ""}">${h.label}</th>`).join("") + "</tr></thead><tbody>" +
    rows.map(r => "<tr>" + r.map((c, i) => `<td class="${headers[i].cls || ""}">${c}</td>`).join("") + "</tr>").join("") + "</tbody>";
}
const riskPill = lvl => `<span class="pill risk-${String(lvl).toLowerCase()}">${esc(lvl)}</span>`;
const statusPill = st => `<span class="pill ${st === "Under Review" ? "review" : st.toLowerCase()}">${esc(st)}</span>`;
const rewardCell = r => `<span class="${r > 0 ? "pos" : r < 0 ? "neg" : ""}">${signed(r)}</span>`;
const anomalyLabel = k => (S.meta && S.meta.anomaly_types[k]) || k || "";

// =============================================================== navigation
async function showPage(name) {
  if (!PAGES[name]) name = "dashboard";
  $$(".page").forEach(p => p.classList.toggle("hidden", p.id !== "page-" + name));
  $$("#nav a").forEach(a => a.classList.toggle("active", a.dataset.page === name));
  $("#page-title").textContent = PAGES[name];
  if (location.hash !== "#" + name) history.replaceState(null, "", "#" + name);
  const loaders = { dashboard: loadDashboard, transactions: loadTransactions, agent: async () => renderSession(), reports: onReportsShown, environment: async () => renderEnvPage() };
  try { await loaders[name](); } catch (e) { showError(e.message); }
}
window.addEventListener("hashchange", () => { if (!S.demo) showPage(location.hash.slice(1)); });

// =============================================================== A. DASHBOARD
async function loadDashboard() {
  const split = $("#dash-split").value, policy = $("#dash-policy").value;
  const [d, cmp, tl] = await Promise.all([
    api(`/api/dashboard?split=${split}&policy=${policy}`), api(`/api/compare?split=${split}`), api("/api/training-log")]);
  $("#dash-scope").textContent = split === "dev"
    ? "Results on the 10 development episodes the RL agent has never seen."
    : "Results on the 10 training episodes the RL agent learned from (optimistic).";
  $("#kpis").innerHTML = [
    ["Total transactions", d.total_transactions, `${d.split_transactions} in the ${split} set`, ""],
    ["Audited transactions", d.audited_transactions, `${d.episodes} episodes · ${esc(d.policy_name)}`, ""],
    ["Suspicious transactions", d.suspicious_flagged, `flagged · ${d.actual_suspicious} truly suspicious`, ""],
    ["Audit accuracy", pct(d.accuracy), `precision ${pct(d.precision)} · recall ${pct(d.recall)}`, "accent"],
    ["Total reward", d.total_reward, `mean ${d.mean_reward} per episode`, "accent"],
  ].map(([t, v, s, c]) => `<div class="kpi ${c}"><small>${t}</small><b>${v}</b><span>${s}</span></div>`).join("");

  const keys = Object.keys(d.per_episode);
  lineChart($("#chart-rewards"), {
    labels: d.episode_ids.map((_, i) => i + 1),
    tooltipTitle: i => "Episode " + d.episode_ids[i],
    xTitle: "Episode", yTitle: "Total reward",
    series: keys.map(k => ({ name: d.policy_names[k], color: COLORS[k], values: d.per_episode[k], width: k === policy ? 3.4 : 1.8 })),
  });
  $("#legend-rewards").innerHTML = keys.map(k => `<span><i style="background:${COLORS[k]}"></i>${esc(d.policy_names[k])}</span>`).join("");

  const log = tl.log;
  lineChart($("#chart-learning"), {
    labels: log.map(l => l.epoch), points: false, xTickEvery: 50, xTitle: "Training epoch", yTitle: "Mean reward / episode",
    tooltipTitle: i => `Epoch ${log[i].epoch} (ε = ${log[i].epsilon})`,
    series: [{ name: "Training reward (explores randomly)", color: COLORS.rl_agent, values: log.map(l => l.mean_reward), width: 2 }],
  });
  $("#learning-note").textContent = `Tabular Q-learning, ${tl.agent.epochs} epochs on 10 training episodes. ` +
    "Reward rises as the agent explores less and exploits what it learned.";

  const best = cmp.best_total_reward;
  table($("#compare-table"),
    [{ label: "Policy" }, { label: "Total reward", cls: "num" }, { label: "Accuracy", cls: "num" }, { label: "Precision", cls: "num" },
      { label: "Recall", cls: "num" }, { label: "Investigations", cls: "num" }, { label: "Doc requests", cls: "num" }],
    cmp.policies.map(p => [
      `<b>${esc(p.name)}</b>${p.policy === best ? ' <span class="pill">best reward</span>' : ""}<br><span class="muted small">${esc(p.description)}</span>`,
      p.total_reward, pct(p.accuracy), pct(p.precision), pct(p.recall), p.investigations, p.doc_requests]));
  $$("#compare-table tbody tr").forEach((tr, i) => tr.classList.toggle("best", cmp.policies[i].policy === best));
  $("#compare-note").textContent = cmp.note;
}

// =============================================================== B. TRANSACTIONS
let searchTimer;
function txEpisodeOptions() {
  const split = $("#tx-split").value;
  const eps = [];
  for (const sp of ["train", "dev"]) if (split === "all" || split === sp) S.meta.episodes[sp].forEach(e => eps.push({ value: e.episode_id, label: e.episode_id }));
  fillSelect($("#tx-episode"), [{ value: "", label: "All episodes" }, ...eps]);
}
async function loadTransactions() {
  const q = new URLSearchParams({
    split: $("#tx-split").value, episode: $("#tx-episode").value, search: $("#tx-search").value,
    category: $("#tx-category").value, risk: $("#tx-risk").value, status: $("#tx-status").value, limit: 2000,
  });
  const r = await api("/api/transactions?" + q);
  S.txRows = r.transactions; S.txTotal = r.total; S.txPage = 1;
  renderTxTable();
}
function renderTxTable() {
  const PAGE = 15, truth = $("#tx-truth").checked;
  const pages = Math.max(1, Math.ceil(S.txRows.length / PAGE));
  S.txPage = Math.min(S.txPage, pages);
  const rows = S.txRows.slice((S.txPage - 1) * PAGE, S.txPage * PAGE);
  const headers = [{ label: "Transaction ID" }, { label: "Date" }, { label: "Amount", cls: "num" }, { label: "Category" },
    { label: "Account" }, { label: "Status" }, { label: "Risk level" }];
  if (truth) headers.push({ label: "Ground truth (synthetic)" });
  table($("#tx-table"), headers, rows.map(t => {
    const r = [`<span class="mono">${esc(t.transaction_id)}</span>`, esc(t.date), money(t.amount), esc(t.category),
      `<span class="mono">${esc(t.account_id)}</span>`, statusPill(t.status), `${riskPill(t.risk_level)} <span class="muted small">${t.risk_score}</span>`];
    if (truth) r.push(t.is_suspicious ? `<span class="pill bad">Suspicious</span> <span class="muted small">${esc(anomalyLabel(t.anomaly_type))}</span>` : '<span class="pill ok">Legitimate</span>');
    return r;
  }), "No transactions match these filters.");
  $("#tx-count").textContent = `${S.txTotal} transactions match`;
  $("#tx-page").textContent = `Page ${S.txPage} / ${pages}`;
  $("#tx-prev").disabled = S.txPage <= 1; $("#tx-next").disabled = S.txPage >= pages;
}

// =============================================================== C. AI AUDIT AGENT
function currentTxHtml(tx, extra) {
  const chips = [];
  if (tx.amount_ratio > 3) chips.push(`${tx.amount_ratio}× category norm`);
  if (tx.near_threshold) chips.push("just below 10,000 limit");
  if (tx.country_mismatch) chips.push(`country ${esc(tx.tx_country)} ≠ ${esc(tx.account_country)}`);
  if (tx.off_hours) chips.push("off-hours");
  if (tx.velocity >= 2) chips.push(`${tx.velocity} tx same day`);
  if (tx.new_account) chips.push(`young account (${tx.account_age_days} d)`);
  if (tx.round_amount) chips.push("round amount");
  const docCls = tx.doc_status === "refused" || tx.doc_status === "missing" ? "bad" : tx.doc_status === "available" || tx.doc_status === "received" ? "ok" : "neutral";
  const invCls = tx.investigation_result === "problem found" ? "bad" : tx.investigation_result === "looks clean" ? "ok" : "neutral";
  return `<div class="tx-head"><span class="mono">${esc(tx.transaction_id)}</span><span class="amount">${money(tx.amount)}</span></div>
    <div class="tx-meta">
      <div><small>Category</small>${esc(tx.category)}</div>
      <div><small>Date / time</small>${esc(tx.date)} ${esc(tx.time)}</div>
      <div><small>Account</small>${esc(tx.account_id)} · ${esc(tx.account_type)}</div>
      <div><small>Risk</small>${riskPill(tx.risk_level)} <span class="muted small">${tx.risk_score}</span></div>
      <div><small>Document</small><span class="pill ${docCls}">${esc(tx.doc_status)}</span></div>
      <div><small>Investigation</small><span class="pill ${invCls}">${esc(tx.investigation_result)}</span></div>
    </div>
    <div class="chips">${chips.length ? chips.map(c => `<span class="chip">${c}</span>`).join("") : '<span class="chip calm">no visible risk indicators</span>'}</div>
    ${extra || ""}`;
}

function stepActionText(s) {
  return s.decision_made ? (s.action_name === "flag" ? "Flagged" : "Approved") : (s.action_name === "investigate" ? "Investigated" : "Requested documents");
}

function renderLog(steps, highlightLast) {
  const rows = steps.map(s => {
    const o = s.outcome ? OUTCOME_TEXT[s.outcome] : null;
    return [s.step, `<span class="mono">${esc(s.transaction_id)}</span>`, `<span class="act ${s.action_name}">${ACTION_TEXT[s.action_name]}</span>`,
      rewardCell(s.reward), o ? `<span class="pill ${o[1]}">${o[0]}</span>` : '<span class="muted small">information step</span>'];
  });
  table($("#ag-log"), [{ label: "#" }, { label: "Transaction" }, { label: "Action" }, { label: "Reward", cls: "num" }, { label: "Result" }],
    rows, "No decisions yet.");
  const wrap = $("#ag-log").parentElement;
  if (highlightLast && rows.length) { $("#ag-log tbody tr:last-child").classList.add("row-new"); wrap.scrollTop = wrap.scrollHeight; }
  const flagged = steps.filter(s => s.action_name === "flag");
  $("#ag-flagged").innerHTML = flagged.length ? flagged.map(s => {
    const t = s.transaction, ok = s.outcome === "true_positive";
    return `<div class="flag-item"><div><span class="mono">${esc(t.transaction_id)}</span> · ${money(t.amount)}<br><span class="muted small">${esc(t.category)} · ${esc(t.account_id)}</span></div>
      <div style="text-align:right">${riskPill(t.risk_level)}<br><span class="pill ${ok ? "ok" : "bad"}">${ok ? "truly suspicious" : "false alarm"}</span></div></div>`;
  }).join("") : "Nothing flagged yet.";
}

function renderStats(step, decided, n, reward, status) {
  $("#ag-decided").textContent = `${decided} / ${n}`;
  $("#ag-steps").textContent = step;
  $("#ag-reward").textContent = reward;
  $("#ag-status").textContent = status;
  $("#ag-progress").style.width = (n ? decided / n * 100 : 0) + "%";
}

function renderResult(st) {
  const box = $("#ag-result");
  if (!st || !st.done) { box.classList.add("hidden"); return; }
  const m = st.summary;
  box.classList.remove("hidden");
  box.innerHTML = `<h3>Final episode result · ${esc(st.episode_id)} · ${esc(st.policy_name)}${st.truncated ? ' <span class="pill bad">cut off</span>' : ' <span class="pill ok">completed</span>'}</h3>
    <div class="result-grid">
      <div><small>Total reward</small><b>${m.total_reward}</b></div>
      <div><small>Accuracy</small><b>${pct(m.accuracy)}</b></div>
      <div><small>Precision</small><b>${pct(m.precision)}</b></div>
      <div><small>Recall</small><b>${pct(m.recall)}</b></div>
      <div><small>Caught / missed</small><b>${m.true_positive} / ${m.false_negative}</b></div>
      <div><small>False alarms</small><b>${m.false_positive}</b></div>
      <div><small>Investigations</small><b>${m.investigations}</b></div>
      <div><small>Doc requests</small><b>${m.doc_requests}</b></div>
    </div>
    <button class="btn primary" id="ag-to-report">View audit report →</button>`;
  $("#ag-to-report").onclick = () => { $("#rp-source").value = "session"; showPage("reports"); };
}

function renderSession() {
  const st = S.session;
  const active = st && st.active;
  const manual = active && st.policy === "manual";
  $("#ag-step").disabled = !(active && !st.done && !manual);
  $("#ag-run").disabled = !(active && !st.done && !manual);
  $("#manual-actions").classList.toggle("hidden", !(manual && !st.done));
  if (!active) {
    renderStats("–", "–", 0, "–", "Not started"); $("#ag-progress").style.width = "0";
    $("#ag-current").innerHTML = "Start an audit episode to see transactions here.";
    $("#ag-current").className = "muted"; renderLog([]); renderResult(null);
    renderEnvPage(); return;
  }
  renderStats(st.step, st.decided, st.n_transactions, st.total_reward,
    st.status === "running" ? "Running" : st.status === "completed" ? "Completed" : "Cut off");
  renderLog(st.steps);
  const cur = $("#ag-current");
  cur.className = "";
  if (st.done) {
    const last = st.last_step;
    cur.innerHTML = `<div class="decision-box">Episode finished. Total reward ${st.total_reward}.</div>` + (last ? `<p class="muted small">Last transaction: ${esc(last.transaction_id)}</p>` : "");
  } else {
    const last = st.last_step;
    const extra = last ? `<div class="decision-box">Last step: ${esc(stepActionText(last))} ${esc(last.transaction_id)} → reward ${signed(last.reward)}</div>` : "";
    cur.innerHTML = currentTxHtml(st.current_transaction, extra);
  }
  renderResult(st);
  if (st.done) { // a freshly finished run becomes the default source of the report page
    const key = `${st.episode_id}|${st.policy}|${st.step}`;
    if (key !== S.runKey) { S.runKey = key; $("#rp-source").value = "session"; toggleReportControls(); }
  }
  renderEnvPage();
}

const startEpisode = guard(async () => {
  S.animToken++; // stop any running animation
  S.session = await post("/api/episode/start", {
    split: $("#ag-split").value, episode_index: Number($("#ag-episode").value), policy: $("#ag-policy").value });
  renderSession();
});
const stepEpisode = guard(async (action) => {
  const r = await post("/api/episode/step", action ? { action } : {});
  S.session = r.state; renderSession();
});

// Run the agent to the end on the server, then replay the steps one by one so people can follow along.
async function runAgent() {
  const token = ++S.animToken;
  const res = await post("/api/episode/run");
  const steps = res.new_steps, final = res.state;
  const before = final.steps.length - steps.length;
  const all = final.steps;
  ["#ag-step", "#ag-run"].forEach(s => $(s).disabled = true);
  const delay = Math.max(25, Math.min(140, 7000 / Math.max(steps.length, 1)));
  let decided = all.slice(0, before).filter(s => s.decision_made).length;
  for (let i = 0; i < steps.length; i++) {
    if (token !== S.animToken) return; // user started a new episode
    const s = steps[i];
    if (s.decision_made) decided++;
    const shown = all.slice(0, before + i + 1);
    renderStats(s.step, decided, final.n_transactions, s.total_reward, "Running");
    $("#ag-current").className = "";
    $("#ag-current").innerHTML = currentTxHtml(s.transaction,
      `<div class="decision-box">${esc(stepActionText(s))} → reward ${signed(s.reward)}${s.outcome ? " · " + OUTCOME_TEXT[s.outcome][0] : ""}</div>`);
    renderLog(shown, true);
    await sleep(delay);
  }
  if (token !== S.animToken) return;
  S.session = final; renderSession();
}
const runAgentSafe = guard(runAgent);

// =============================================================== D. AUDIT REPORTS
function reportQuery() {
  const src = $("#rp-source").value;
  if (src === "session") return "source=session";
  return `source=episode&split=${$("#rp-split").value}&episode_index=${$("#rp-episode").value}&policy=${$("#rp-policy").value}`;
}
function toggleReportControls() { $("#rp-custom").classList.toggle("hidden", $("#rp-source").value !== "episode"); }

async function onReportsShown() {
  const hasRun = S.session && S.session.done;
  // nothing has been run yet -> show a default episode instead of an empty page
  if (!hasRun && $("#rp-source").value === "session") $("#rp-source").value = "episode";
  toggleReportControls();
  await loadReport();
}
async function loadReport() {
  const q = reportQuery();
  $("#rp-json").onclick = () => { location.href = `/api/report/export?format=json&${q}`; };
  $("#rp-csv").onclick = () => { location.href = `/api/report/export?format=csv&${q}`; };
  let r;
  try { r = await api("/api/report?" + q); }
  catch (e) {
    $("#rp-body").classList.add("hidden");
    $("#rp-empty").classList.remove("hidden");
    $("#rp-empty").textContent = e.message;
    return;
  }
  $("#rp-empty").classList.add("hidden"); $("#rp-body").classList.remove("hidden");
  const s = r.summary, m = s.metrics;
  $("#rp-kpis").innerHTML = [
    ["Risk rating", `<span class="rating ${s.risk_rating}" style="font-size:22px;padding:2px 12px">${s.risk_rating}</span>`, `${s.flagged_amount_pct}% of amount flagged`, ""],
    ["Flagged", `${s.flagged} / ${s.transactions}`, `${s.approved} approved`, ""],
    ["Flagged amount", money(s.flagged_amount), `of ${money(s.total_amount)}`, ""],
    ["Accuracy", pct(m.accuracy), `precision ${pct(m.precision)} · recall ${pct(m.recall)}`, "accent"],
    ["Total reward", m.total_reward, `${m.investigations} investigations`, "accent"],
  ].map(([t, v, sub, c]) => `<div class="kpi ${c}"><small>${t}</small><b>${v}</b><span>${sub}</span></div>`).join("");
  $("#rp-narrative").textContent = s.narrative;
  $("#rp-meta").textContent = `Episode ${r.meta.episode_id} · ${r.meta.policy_name} · generated ${r.meta.generated_at} · ${r.meta.data_note}`;

  table($("#rp-anomalies"),
    [{ label: "Transaction" }, { label: "Amount", cls: "num" }, { label: "Category" }, { label: "Risk" }, { label: "Why flagged" }, { label: "Check vs. synthetic truth" }],
    r.anomalies.map(a => [`<span class="mono">${esc(a.transaction_id)}</span><br><span class="muted small">${esc(a.date)}</span>`, money(a.amount), esc(a.category),
      riskPill(a.risk_level),
      a.reasons.length ? `<ul class="reasons">${a.reasons.map(x => `<li>${esc(x)}</li>`).join("")}</ul>` : '<span class="muted small">no single indicator</span>',
      a.correct ? `<span class="pill ok">correct</span><br><span class="muted small">${esc(a.anomaly_type_truth)}</span>` : '<span class="pill bad">false alarm</span>']),
    "The agent flagged nothing in this episode.");
  table($("#rp-incorrect"),
    [{ label: "Transaction" }, { label: "Ledger", cls: "num" }, { label: "Invoice", cls: "num" }, { label: "Diff", cls: "num" }, { label: "Caught?" }],
    r.incorrect_transactions.map(t => [`<span class="mono">${esc(t.transaction_id)}</span>`, money(t.ledger_amount), money(t.invoice_amount),
      `${signed(t.difference_pct)}%`, t.caught ? '<span class="pill ok">flagged</span>' : '<span class="pill bad">not flagged</span>']),
    "No inconsistent records in this episode.");
  table($("#rp-missing"),
    [{ label: "Transaction" }, { label: "Amount", cls: "num" }, { label: "Docs requested?" }, { label: "Outcome" }, { label: "Decision" }],
    r.missing_documents.map(t => [`<span class="mono">${esc(t.transaction_id)}</span>`, money(t.amount), t.requested ? "yes" : "no",
      `<span class="pill ${t.doc_outcome === "received" ? "ok" : t.doc_outcome === "refused" ? "bad" : "neutral"}">${esc(t.doc_outcome)}</span>`, esc(t.decision)]),
    "No missing documents in this episode.");
  const ra = r.risk_assessment, maxAmt = Math.max(1, ...ra.by_category.map(c => c.flagged_amount));
  $("#rp-risk").innerHTML = `<span class="rating ${ra.rating}">${ra.rating} risk</span>
    <p class="muted small">${esc(ra.rule)}</p><h4>Flagged amount by category</h4>` +
    ra.by_category.map(c => `<div class="bar-row"><span>${esc(c.category)}</span><div class="bar"><i style="width:${c.flagged_amount / maxAmt * 100}%"></i></div><span class="num">${c.flagged} flagged</span></div>`).join("");
  table($("#rp-missed"),
    [{ label: "Transaction" }, { label: "Amount", cls: "num" }, { label: "Category" }, { label: "What it really was" }],
    r.missed_suspicious_transactions.map(t => [`<span class="mono">${esc(t.transaction_id)}</span>`, money(t.amount), esc(t.category), esc(t.anomaly_type_truth)]),
    "No suspicious transaction was missed.");
}

// =============================================================== E. RL ENVIRONMENT
function renderEnvPage() {
  const st = S.session, active = st && st.active;
  const last = active ? st.last_step : null;
  $("#env-state").innerHTML = [
    ["Current episode", active ? st.episode_id : "–"],
    ["Current step", active ? st.step : "–"],
    ["Agent action", last ? ACTION_TEXT[last.action_name] : "–"],
    ["Reward (last step)", last ? signed(last.reward) : "–"],
    ["Total reward", active ? st.total_reward : "–"],
    ["Episode status", active ? (st.status === "running" ? "Running" : st.status === "completed" ? "Terminated (completed)" : "Truncated") : "Not started"],
  ].map(([k, v]) => `<div><small>${k}</small><b>${esc(v)}</b></div>`).join("");
  $("#env-step").disabled = !(active && !st.done && st.policy !== "manual");
  $("#env-start").textContent = active ? "Restart episode" : "Start episode (RL agent, dev-00)";
  const obsBox = $("#env-obs");
  const named = active ? (st.current_observation || (last && last.observation_named)) : null;
  if (!named) { obsBox.className = "muted"; obsBox.textContent = "No active episode."; return; }
  obsBox.className = "";
  const desc = Object.fromEntries(S.meta.observation_fields.map(f => [f.name, f.description]));
  obsBox.innerHTML = (st.done ? '<p class="muted small">Episode finished – showing the last observation the agent saw.</p>' : "") +
    Object.entries(named).map(([k, v]) => `<div class="obs-row" title="${esc(k)}"><span>${esc(desc[k] || k)}</span><div class="bar"><i style="width:${v * 100}%"></i></div><span class="mono">${v.toFixed(2)}</span></div>`).join("");
}

function initEnvStatic() {
  table($("#env-rewards"), [{ label: "Situation" }, { label: "Reward", cls: "num" }],
    S.meta.rewards.map(r => [esc(r.description), rewardCell(r.value)]));
  table($("#env-fields"), [{ label: "#" }, { label: "Field" }, { label: "Meaning" }],
    S.meta.observation_fields.map(f => [f.index, `<span class="mono">${esc(f.name)}</span>`, esc(f.description)]));
}

// =============================================================== DEMO MODE
function caption(text) { $("#demo-text").textContent = text; $("#demo-caption").classList.remove("hidden"); }
function stopDemo(msg) {
  if (S.demo) S.demo.cancelled = true;
  S.demo = null; S.animToken++;
  $("#demo-caption").classList.add("hidden");
  $$(".highlight").forEach(e => e.classList.remove("highlight"));
  $("#demo-btn").textContent = "▶ Start demo mode";
  if (msg) showError(msg);
}
async function runDemo() {
  if (S.demo) { stopDemo(); return; }
  const demo = S.demo = { cancelled: false };
  $("#demo-btn").textContent = "■ Stop demo";
  const alive = () => !demo.cancelled;
  const wait = async ms => { const end = Date.now() + ms; while (alive() && Date.now() < end) await sleep(100); };
  const focus = sel => { $$(".highlight").forEach(e => e.classList.remove("highlight")); const el = $(sel); if (el) { el.classList.add("highlight"); el.scrollIntoView({ behavior: "smooth", block: "center" }); } };
  try {
    // 1) dashboard
    $("#dash-policy").value = "rl_agent"; $("#dash-split").value = "dev";
    await showPage("dashboard"); window.scrollTo({ top: 0, behavior: "smooth" });
    caption("1/6 · Dashboard: KPIs, agent performance and learning curve"); focus("#kpis"); await wait(6500);
    if (!alive()) return;
    // 2) transactions
    $("#tx-split").value = "dev"; txEpisodeOptions(); $("#tx-risk").value = ""; $("#tx-search").value = "";
    await showPage("transactions"); window.scrollTo({ top: 0 });
    caption("2/6 · Transactions: synthetic data with search and filters"); focus("#tx-table"); await wait(4500);
    if (!alive()) return;
    $("#tx-risk").value = "High"; await loadTransactions();
    caption("   … filtered to High-risk transactions"); await wait(3500);
    if (!alive()) return;
    $("#tx-risk").value = ""; $("#tx-split").value = "all"; txEpisodeOptions();
    // 3) run the agent
    $("#ag-split").value = "dev"; fillEpisodeSelect($("#ag-episode"), "dev"); $("#ag-episode").value = "0"; $("#ag-policy").value = "rl_agent";
    await showPage("agent"); window.scrollTo({ top: 0 });
    caption("3/6 · Running the RL audit agent on episode dev-00…");
    await startEpisode(); await wait(1200);
    if (!alive()) return;
    await runAgent();
    if (!alive()) return;
    // 4) flagged
    caption("4/6 · Suspicious transactions detected (flagged) and the final episode result");
    focus("#ag-flagged"); await wait(4500);
    if (!alive()) return;
    focus("#ag-result"); await wait(3500);
    if (!alive()) return;
    // 5) comparison
    await showPage("dashboard");
    caption("5/6 · Comparing the four policies on unseen episodes"); focus("#compare-card"); await wait(7500);
    if (!alive()) return;
    // 6) report
    $("#rp-source").value = "session";
    await showPage("reports"); window.scrollTo({ top: 0 });
    caption("6/6 · Final audit report, exportable as JSON or CSV"); focus("#rp-narrative"); await wait(4500);
    if (!alive()) return;
    focus("#rp-anomalies"); await wait(5000);
    if (!alive()) return;
    caption("Demo finished. Thank you!"); await wait(2500);
  } catch (e) {
    showError("Demo stopped: " + e.message);
  } finally {
    stopDemo();
  }
}

// =============================================================== INIT
function fillEpisodeSelect(sel, split) {
  fillSelect(sel, S.meta.episodes[split].map(e => ({ value: e.episode_index, label: `${e.episode_id} (${e.n_transactions} tx)` })));
}

async function init() {
  try {
    await api("/api/health");
    setStatus(true);
    S.meta = await api("/api/meta");
  } catch (e) {
    setStatus(false); showError(e.message); return;
  }
  const pol = S.meta.policies.map(p => ({ value: p.key, label: p.name }));
  fillSelect($("#dash-policy"), pol, "rl_agent");
  fillSelect($("#ag-policy"), [...pol, { value: "manual", label: "Me (manual play)" }], "rl_agent");
  fillSelect($("#rp-policy"), pol, "rl_agent");
  fillEpisodeSelect($("#ag-episode"), "dev"); fillEpisodeSelect($("#rp-episode"), "dev");
  fillSelect($("#tx-category"), [{ value: "", label: "All categories" }, ...S.meta.categories.map(c => ({ value: c, label: c }))]);
  txEpisodeOptions(); initEnvStatic();

  // dashboard
  $("#dash-policy").onchange = $("#dash-split").onchange = guard(loadDashboard);
  // transactions
  $("#tx-split").onchange = guard(async () => { txEpisodeOptions(); await loadTransactions(); });
  ["#tx-episode", "#tx-category", "#tx-risk", "#tx-status"].forEach(s => $(s).onchange = guard(loadTransactions));
  $("#tx-search").oninput = () => { clearTimeout(searchTimer); searchTimer = setTimeout(guard(loadTransactions), 250); };
  $("#tx-truth").onchange = renderTxTable;
  $("#tx-prev").onclick = () => { S.txPage--; renderTxTable(); };
  $("#tx-next").onclick = () => { S.txPage++; renderTxTable(); };
  // agent
  $("#ag-split").onchange = () => fillEpisodeSelect($("#ag-episode"), $("#ag-split").value);
  $("#ag-start").onclick = startEpisode;
  $("#ag-step").onclick = () => stepEpisode();
  $("#ag-run").onclick = runAgentSafe;
  $$("#manual-actions [data-action]").forEach(b => b.onclick = () => stepEpisode(b.dataset.action));
  // reports
  $("#rp-source").onchange = () => { toggleReportControls(); guard(loadReport)(); };
  $("#rp-split").onchange = () => fillEpisodeSelect($("#rp-episode"), $("#rp-split").value);
  $("#rp-generate").onclick = guard(loadReport);
  // environment
  $("#env-start").onclick = guard(async () => {
    if (!S.session || !S.session.active) { $("#ag-split").value = "dev"; fillEpisodeSelect($("#ag-episode"), "dev"); $("#ag-episode").value = "0"; $("#ag-policy").value = "rl_agent"; }
    await startEpisode();
  });
  $("#env-step").onclick = () => stepEpisode();
  $("#env-refresh").onclick = guard(async () => { S.session = await api("/api/episode/current"); renderSession(); });
  // demo
  $("#demo-btn").onclick = runDemo;
  $("#demo-stop").onclick = () => stopDemo();

  S.session = await api("/api/episode/current");
  await showPage(location.hash.slice(1) || "dashboard");
}
init();
