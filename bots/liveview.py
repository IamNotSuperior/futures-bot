"""A read-only page that follows a walk-forward while it runs.

    venv\\Scripts\\python.exe bots\\liveview.py            # http://127.0.0.1:8790
    venv\\Scripts\\python.exe bots\\liveview.py --port 8791

Reads the event stream a runner writes with ``--live`` (``backtests/live.py``)
and shows the stages advancing, the equity curve drawn trade by trade from the
priced stream once it exists, the fold table, and the outcome. Every route is
``GET``; the page has no form, no button that changes anything, and this
module imports no runner. It is a window on a run, not a lever.

A walk-forward here generates every signal in one call, prices the stream in
one call and scores the folds in one call, so the stages *are* the run's real
granularity. The equity animation is a replay of the finished stream and the
page says so.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _folder in ("backtests",):
    _p = str(PROJECT_ROOT / _folder)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import json  # noqa: E402
from datetime import datetime  # noqa: E402

import pandas as pd  # noqa: E402
from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse  # noqa: E402

import live  # noqa: E402
import runners  # noqa: E402  (its table of saved files only; nothing here runs)

DEFAULT_HOST = "127.0.0.1"
#: The desk feed binds 8787; this stays clear of it.
DEFAULT_PORT = 8790

SAVED_BASES = ("standard", "comparable")


def _sibling(results: Path, filename: str | None) -> str | None:
    """The halt-OFF stream saved beside a guarded one, when there is one."""
    if not filename or not filename.endswith(".csv"):
        return None
    candidate = filename[:-4] + "_nohalt.csv"
    return candidate if (results / candidate).exists() else None


def saved_catalogue(results: Path, table: dict, registry) -> list[dict]:
    """Every registry name with the saved files its verdict quotes.

    Names in ``runners.RUNNERS`` use that table's files and note; anything
    else follows the generated runner's fixed naming. A name whose files are
    missing is listed as unavailable rather than hidden, so a regenerable
    result reads as "not on disk" instead of "never existed".
    """
    rows: list[dict] = []
    for name in registry.names():
        record = registry.get(name)
        if name in table:
            runner = table[name]
            trades, folds, note = runner.oos_csv, runner.walkforward_csv, runner.note
        else:
            trades = f"{name}_trades.csv" if (results / f"{name}_trades.csv").exists() else None
            folds = f"{name}_folds.csv" if (results / f"{name}_folds.csv").exists() else None
            note = "generated strategy; the verdict's guarded stream at the base case" if trades else ""
        path = results / trades if trades else None
        available = bool(path is not None and path.exists())
        produced = (datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="minutes")
                    if available else None)
        rows.append({
            "name": name,
            "entry": record.hypothesis_entry,
            "status": record.status,
            "note": note or "",
            "trades_file": trades,
            "folds_file": folds if (folds and (results / folds).exists()) else None,
            "has_comparable": _sibling(results, trades) is not None,
            "available": available,
            "produced": produced,
        })
    return rows


def _read_results_csv(results: Path, filename: str | None) -> pd.DataFrame | None:
    """A CSV from the results directory and nowhere else."""
    if not filename:
        return None
    results = results.resolve()
    path = (results / filename).resolve()
    if path.parent != results or not path.exists():
        return None
    return pd.read_csv(path)


def build_app(directory: Path | None = None, results: Path | None = None,
              table: dict | None = None, registry_loader=None) -> FastAPI:
    app = FastAPI(title="live backtest view", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.directory = Path(directory) if directory is not None else live.LIVE_DIR
    app.state.results = Path(results) if results is not None else runners.RESULTS
    app.state.table = table if table is not None else runners.RUNNERS
    app.state.registry_loader = registry_loader if registry_loader is not None else runners.registry

    def _dir() -> Path:
        return app.state.directory

    def _catalogue() -> dict[str, dict]:
        rows = saved_catalogue(app.state.results, app.state.table, app.state.registry_loader())
        return {r["name"]: r for r in rows}

    def _saved(name: str) -> dict:
        entry = _catalogue().get(name)
        if entry is None:
            raise HTTPException(status_code=404)
        return entry

    @app.get("/saved")
    def saved() -> list[dict]:
        return list(_catalogue().values())

    @app.get("/saved/{name}/trades")
    def saved_trades(name: str, basis: str = "standard") -> JSONResponse:
        entry = _saved(name)
        if basis not in SAVED_BASES:
            raise HTTPException(status_code=404)
        filename = entry["trades_file"] if basis == "standard" else _sibling(app.state.results, entry["trades_file"])
        frame = _read_results_csv(app.state.results, filename)
        return JSONResponse([] if frame is None else live.frame_points(frame))

    @app.get("/saved/{name}/folds")
    def saved_folds(name: str) -> JSONResponse:
        entry = _saved(name)
        frame = _read_results_csv(app.state.results, entry["folds_file"])
        return JSONResponse([] if frame is None else live.frame_rows(frame))

    @app.get("/data")
    def data_sources() -> list[dict]:
        """The data-check reports ``data/topstep.py --compare`` writes."""
        results: Path = app.state.results
        out: list[dict] = []
        for path in sorted(results.glob("data_check_*.json")) if results.exists() else []:
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(report, dict):
                report["file"] = path.name
                out.append(report)
        return out

    @app.get("/", response_class=HTMLResponse)
    def page() -> str:
        return PAGE

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    @app.get("/runs")
    def runs() -> list[dict]:
        return live.list_runs(_dir())

    def _name(value: str) -> str:
        # Only a bad name is a 404. Anything else that goes wrong surfaces
        # as an error rather than masquerading as "not found".
        try:
            return live.validate_name(value)
        except ValueError:
            raise HTTPException(status_code=404) from None

    @app.get("/runs/{name}/events")
    def events(name: str, after: int = Query(0, ge=0)) -> dict:
        return live.read_events(_name(name), after=after, directory=_dir())

    @app.get("/runs/{name}/trades")
    def trades(name: str, basis: str = "standard") -> JSONResponse:
        return JSONResponse(live.read_trades(_name(name), basis=_name(basis), directory=_dir()))

    @app.get("/runs/{name}/folds")
    def folds(name: str) -> JSONResponse:
        return JSONResponse(live.read_folds(_name(name), directory=_dir()))

    return app


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Live backtest view</title>
<style>
  :root {
    --bg: #f6f5f1; --panel: #ffffff; --ink: #1d1c1a; --muted: #6b6860;
    --line: #dedbd3; --accent: #1f5f8b; --good: #2c7a4b; --bad: #a8352b;
    --warn: #9a6b12; --mono: ui-monospace, Menlo, Consolas, monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #17181a; --panel: #1f2124; --ink: #e8e6e1; --muted: #9a978f;
            --line: #33363b; --accent: #7fb3d9; --good: #6cc48f; --bad: #e07a70; --warn: #d9a441; }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink);
         font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
  header { display: flex; align-items: baseline; gap: 16px; padding: 14px 20px;
           border-bottom: 1px solid var(--line); background: var(--panel); flex-wrap: wrap; }
  header h1 { font-size: 16px; margin: 0; font-weight: 600; letter-spacing: .01em; }
  header .note { color: var(--muted); font-size: 12px; }
  header select { font: inherit; padding: 4px 8px; background: var(--panel); color: var(--ink);
                  border: 1px solid var(--line); border-radius: 6px; }
  main { display: grid; grid-template-columns: 320px 1fr; gap: 16px; padding: 16px 20px; }
  @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
  section { background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
            padding: 14px 16px; min-width: 0; }
  section h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .08em;
               color: var(--muted); margin: 0 0 10px; font-weight: 600; }
  #banner { grid-column: 1 / -1; display: flex; gap: 14px; align-items: center; }
  .pill { font-family: var(--mono); font-size: 12px; padding: 3px 10px; border-radius: 999px;
          border: 1px solid var(--line); color: var(--muted); }
  .pill.running { color: var(--accent); border-color: var(--accent); }
  .pill.good { color: var(--good); border-color: var(--good); }
  .pill.bad { color: var(--bad); border-color: var(--bad); }
  .pill.warn { color: var(--warn); border-color: var(--warn); }
  #runner { font-family: var(--mono); font-size: 13px; }
  ol.stages { list-style: none; margin: 0; padding: 0; }
  ol.stages li { display: grid; grid-template-columns: 14px 1fr auto; gap: 10px;
                 padding: 6px 0; border-top: 1px solid var(--line); align-items: start; }
  ol.stages li:first-child { border-top: 0; }
  ol.stages .dot { width: 10px; height: 10px; border-radius: 50%; margin-top: 5px;
                   background: var(--line); }
  ol.stages li.done .dot { background: var(--good); }
  ol.stages li.current .dot { background: var(--accent); animation: pulse 1.2s infinite; }
  ol.stages li.error .dot { background: var(--bad); }
  ol.stages .msg { font-family: var(--mono); font-size: 12px; word-break: break-word; }
  ol.stages .dur { font-family: var(--mono); font-size: 11px; color: var(--muted); white-space: nowrap; }
  @keyframes pulse { 0%,100% { opacity: 1 } 50% { opacity: .35 } }
  #equity { width: 100%; height: 320px; display: block; }
  .legend { display: flex; gap: 14px; align-items: center; color: var(--muted); font-size: 12px;
            margin-top: 6px; flex-wrap: wrap; }
  .legend button { font: inherit; font-size: 12px; padding: 3px 9px; border-radius: 6px;
                   border: 1px solid var(--line); background: transparent; color: var(--ink); cursor: pointer; }
  .legend button.on { border-color: var(--accent); color: var(--accent); }
  table { border-collapse: collapse; width: 100%; font-family: var(--mono); font-size: 12px; }
  th, td { text-align: right; padding: 5px 8px; border-top: 1px solid var(--line); white-space: nowrap; }
  th { color: var(--muted); font-weight: 600; border-top: 0; }
  td:first-child, th:first-child { text-align: left; }
  td.pos { color: var(--good); } td.neg { color: var(--bad); }
  .empty { color: var(--muted); font-size: 12px; }
  .scroll { overflow-x: auto; }
</style>
</head>
<body>
<header>
  <h1>Live backtest view</h1>
  <select id="runs" aria-label="run"></select>
  <span class="note">read-only &middot; live runs from <code>backtests/results/live/</code>, saved verdicts from <code>backtests/results/</code> &middot; polls once a second</span>
</header>
<main>
  <section id="banner">
    <span id="status" class="pill">no run</span>
    <span id="runner"></span>
    <span id="started" class="note"></span>
  </section>
  <section>
    <h2>Stages</h2>
    <ol class="stages" id="stages"><li class="empty">Start a runner with <code>--live</code>.</li></ol>
  </section>
  <section>
    <h2>Equity, net of costs</h2>
    <svg id="equity" viewBox="0 0 800 320" preserveAspectRatio="none" role="img" aria-label="equity curve"></svg>
    <div class="legend">
      <span id="eqnote" class="empty">The curve appears when the priced stream exists. It is a replay of the finished stream, drawn trade by trade; the stages are the run's real granularity.</span>
      <button id="b-standard" class="on" data-basis="standard">standard (both guards)</button>
      <button id="b-comparable" data-basis="comparable">comparable (halt off)</button>
      <span id="eqsum"></span>
    </div>
  </section>
  <section style="grid-column: 1 / -1">
    <h2>Folds</h2>
    <div class="scroll"><table id="folds"><tbody><tr><td class="empty">No fold table yet.</td></tr></tbody></table></div>
  </section>
  <section style="grid-column: 1 / -1">
    <h2>Data sources</h2>
    <div id="data" class="empty">Databento is the backtest source. Run <code>data\topstep.py --compare</code> to check a second feed against it; reports appear here.</div>
  </section>
</main>
<script>
(function () {
  const $ = (id) => document.getElementById(id);
  const state = { run: null, saved: null, cursor: 0, events: [], basis: "standard", have: {}, drawn: null, anim: null, started: null };
  const base = () => state.saved ? "/saved/" + state.run : "/runs/" + state.run;

  function fmtMoney(x) { const s = (x < 0 ? "-" : "") + "$" + Math.abs(x).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}); return s; }
  function secs(a, b) { const d = (new Date(b) - new Date(a)) / 1000; return isFinite(d) ? d : 0; }
  function fmtDur(s) { if (s < 60) return s.toFixed(1) + "s"; const m = Math.floor(s / 60); return m + "m " + Math.round(s - 60 * m) + "s"; }

  async function getJSON(url) { const r = await fetch(url, {cache: "no-store"}); if (!r.ok) throw new Error(r.status); return r.json(); }

  let savedRows = [];

  async function refreshRuns() {
    let runs, saved;
    try { [runs, saved] = await Promise.all([getJSON("/runs"), getJSON("/saved")]); }
    catch (e) { return; }  // server away; try again next tick
    savedRows = saved;
    const sel = $("runs");
    const current = sel.value;
    sel.innerHTML = "";
    const live = document.createElement("optgroup"); live.label = "live runs";
    if (!runs.length) { const o = document.createElement("option"); o.disabled = true; o.textContent = "(no live runs yet)"; live.appendChild(o); }
    for (const r of runs) {
      const o = document.createElement("option");
      o.value = r.name; o.textContent = r.name + " · " + r.status + " · " + r.runner;
      live.appendChild(o);
    }
    sel.appendChild(live);
    const group = document.createElement("optgroup"); group.label = "saved verdicts";
    for (const s of saved) {
      const o = document.createElement("option");
      o.value = "saved:" + s.name;
      o.textContent = s.name + " · entry " + s.entry + " · " + s.status + (s.available ? "" : " · not on disk");
      o.disabled = !s.available;
      group.appendChild(o);
    }
    sel.appendChild(group);
    const values = Array.from(sel.options).filter(o => !o.disabled).map(o => o.value);
    if (!current || !values.includes(current)) {
      const running = runs.find(r => r.status === "running");
      const first = running ? running.name : runs.length ? runs[0].name
                  : values.find(v => v.startsWith("saved:"));
      if (first) selectRun(first);
    } else {
      sel.value = current;
    }
  }

  function selectRun(value) {
    if (state.anim) cancelAnimationFrame(state.anim);
    const isSaved = value.startsWith("saved:");
    const name = isSaved ? value.slice(6) : value;
    state.run = name; state.cursor = 0; state.events = []; state.have = {}; state.drawn = null; state.started = null;
    state.saved = isSaved ? (savedRows.find(s => s.name === name) || {name}) : null;
    $("runs").value = value;
    $("stages").innerHTML = "";
    $("equity").innerHTML = "";
    $("eqsum").textContent = "";
    $("folds").innerHTML = "<tbody><tr><td class='empty'>No fold table yet.</td></tr></tbody>";
    if (isSaved) renderSaved(); else poll();
  }

  function renderSaved() {
    const s = state.saved;
    const st = $("status");
    st.className = "pill";
    st.textContent = (s.status || "saved").toUpperCase();
    st.classList.add(/accepted|paper|live/.test(s.status) ? "good" : /rejected/.test(s.status) ? "bad" : "warn");
    $("runner").textContent = "entry " + s.entry + (s.note ? " · " + s.note : "");
    $("started").textContent = s.produced ? "saved result, produced " + s.produced : "saved result";
    const list = $("stages");
    list.innerHTML = "";
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "No stages: this is a finished verdict read from " + (s.trades_file || "no stream") +
      (s.folds_file ? " and " + s.folds_file : " (no fold table saved)") + ". Nothing is running.";
    list.appendChild(li);
    state.have = { standard: !!s.available, comparable: !!s.has_comparable };
    for (const b of ["standard", "comparable"]) {
      $("b-" + b).disabled = !state.have[b];
      $("b-" + b).style.opacity = state.have[b] ? 1 : .5;
    }
    if (!state.have[state.basis]) { state.basis = "standard"; $("b-standard").classList.add("on"); $("b-comparable").classList.remove("on"); }
    loadTrades();
    loadFolds();
  }

  async function poll() {
    if (!state.run || state.saved) return;
    let out;
    try { out = await getJSON("/runs/" + state.run + "/events?after=" + state.cursor); }
    catch (e) { return; }
    if (out.run !== state.run) return;
    // A runner re-run with the same name truncates the file and starts over.
    // The start time changing, or the file shrinking, means a fresh run:
    // drop everything and read it from the beginning.
    if ((state.started && out.started && out.started !== state.started) || out.next < state.cursor) {
      const name = state.run;
      selectRun(name);
      return;
    }
    if (out.started) state.started = out.started;
    if (out.events.length) {
      state.events = state.events.concat(out.events);
      state.cursor = out.next;
      render();
      for (const e of out.events) {
        if (e.event === "trades") { state.have[e.basis] = true; if (e.basis === state.basis) loadTrades(); }
        if (e.event === "folds") loadFolds();
      }
    }
  }

  function render() {
    const ev = state.events;
    if (!ev.length) return;
    const start = ev[0];
    const last = ev[ev.length - 1];
    $("runner").textContent = start.runner || "";
    $("started").textContent = start.t ? "started " + start.t : "";
    const st = $("status");
    st.className = "pill";
    if (last.event === "done") {
      st.textContent = last.status;
      st.classList.add(/ACCEPTED|REPRODUCED|PASSED/.test(last.status) ? "good" : /REJECTED|DIVERGED|FAILED/.test(last.status) ? "bad" : "warn");
    } else if (last.event === "error") { st.textContent = "error"; st.classList.add("bad"); }
    else { st.textContent = "running"; st.classList.add("running"); }

    const stages = ev.filter(e => e.event === "stage" || e.event === "error");
    const list = $("stages");
    list.innerHTML = "";
    const finished = last.event === "done" || last.event === "error";
    stages.forEach((e, i) => {
      const li = document.createElement("li");
      const next = i + 1 < stages.length ? stages[i + 1].t : (finished ? last.t : null);
      const dur = next ? fmtDur(secs(e.t, next)) : fmtDur(secs(e.t, new Date().toISOString()));
      const isLast = i === stages.length - 1;
      li.className = e.event === "error" ? "error" : (isLast && !finished) ? "current" : "done";
      li.innerHTML = "<span class='dot'></span><span class='msg'></span><span class='dur'></span>";
      li.querySelector(".msg").textContent = e.message || "";
      li.querySelector(".dur").textContent = dur;
      list.appendChild(li);
    });
    if (!stages.length) list.innerHTML = "<li class='empty'>Waiting for the first stage.</li>";
    for (const b of ["standard", "comparable"]) {
      $("b-" + b).disabled = !state.have[b];
      $("b-" + b).style.opacity = state.have[b] ? 1 : .5;
    }
  }

  async function loadTrades() {
    let pts;
    try { pts = await getJSON(base() + "/trades?basis=" + state.basis); } catch (e) { return; }
    draw(pts);
  }

  function draw(pts) {
    const svg = $("equity");
    svg.innerHTML = "";
    if (state.anim) cancelAnimationFrame(state.anim);
    if (!pts.length) { $("eqsum").textContent = ""; return; }
    const W = 800, H = 320, padL = 8, padR = 8, padT = 12, padB = 12;
    const ys = pts.map(p => p.cum_pnl).concat([0]);
    let lo = Math.min(...ys), hi = Math.max(...ys);
    if (hi === lo) { hi += 1; lo -= 1; }
    const x = (i) => padL + (i / Math.max(1, pts.length - 1)) * (W - padL - padR);
    const y = (v) => padT + (1 - (v - lo) / (hi - lo)) * (H - padT - padB);
    const ns = "http://www.w3.org/2000/svg";
    const zero = document.createElementNS(ns, "line");
    zero.setAttribute("x1", padL); zero.setAttribute("x2", W - padR);
    zero.setAttribute("y1", y(0)); zero.setAttribute("y2", y(0));
    zero.setAttribute("stroke", "var(--line)"); zero.setAttribute("stroke-dasharray", "4 4");
    svg.appendChild(zero);
    const path = document.createElementNS(ns, "path");
    path.setAttribute("fill", "none"); path.setAttribute("stroke", "var(--accent)");
    path.setAttribute("stroke-width", "1.6"); path.setAttribute("vector-effect", "non-scaling-stroke");
    svg.appendChild(path);
    const head = document.createElementNS(ns, "circle");
    head.setAttribute("r", "3.5"); head.setAttribute("fill", "var(--accent)");
    svg.appendChild(head);
    let n = 0;
    const t0 = performance.now();
    const seconds = Math.min(6, Math.max(2, pts.length / 60));  // wall-clock, so a throttled tab still finishes
    function step() {
      n = Math.max(1, Math.min(pts.length, Math.round(pts.length * (performance.now() - t0) / (seconds * 1000))));
      let d = "";
      for (let i = 0; i < n; i++) d += (i ? "L" : "M") + x(i).toFixed(1) + " " + y(pts[i].cum_pnl).toFixed(1);
      path.setAttribute("d", d);
      const lastP = pts[n - 1];
      head.setAttribute("cx", x(n - 1)); head.setAttribute("cy", y(lastP.cum_pnl));
      head.setAttribute("fill", lastP.cum_pnl >= 0 ? "var(--good)" : "var(--bad)");
      $("eqsum").textContent = "trade " + n + " of " + pts.length + " · " + fmtMoney(lastP.cum_pnl) +
        (n === pts.length ? " · peak-to-trough " + fmtMoney(drawdown(pts)) : "");
      if (n < pts.length) state.anim = requestAnimationFrame(step);
    }
    step();
  }

  function drawdown(pts) {
    let peak = 0, worst = 0;
    for (const p of pts) { peak = Math.max(peak, p.cum_pnl); worst = Math.min(worst, p.cum_pnl - peak); }
    return worst;
  }

  async function loadFolds() {
    let rows;
    try { rows = await getJSON(base() + "/folds"); } catch (e) { return; }
    const table = $("folds");
    if (!rows.length) return;
    const cols = Object.keys(rows[0]);
    const thead = "<thead><tr>" + cols.map(c => "<th>" + c.replace(/^test_/, "").replace(/_/g, " ") + "</th>").join("") + "</tr></thead>";
    const body = rows.map(r => "<tr>" + cols.map(c => {
      const v = r[c];
      let cls = "";
      if (c === "test_net_pnl" && typeof v === "number") cls = v > 0 ? "pos" : v < 0 ? "neg" : "";
      let text = v;
      if (v === null || v === undefined) text = "–";  // a halted year has no Sharpe
      else if (typeof v === "number" && !Number.isInteger(v)) text = v.toFixed(c.includes("prob") ? 4 : 2);
      return "<td class='" + cls + "'>" + text + "</td>";
    }).join("") + "</tr>").join("");
    table.innerHTML = thead + "<tbody>" + body + "</tbody>";
  }

  async function loadData() {
    let reports;
    try { reports = await getJSON("/data"); } catch (e) { return; }
    if (!reports.length) return;
    const el = $("data");
    el.className = "";
    el.innerHTML = "";
    for (const r of reports) {
      const c = (r.columns || {}).close || {};
      const days = (r.differing_days || []);
      const box = document.createElement("div");
      box.style.marginBottom = "12px";
      const head = document.createElement("div");
      head.innerHTML = "<strong></strong> <span class='note'></span>";
      head.querySelector("strong").textContent = (r.symbol || "?") + " · " + (r.source || "second feed");
      head.querySelector(".note").textContent = " · " + (r.topstep_file || "") + " vs " + (r.cache_file || "") +
        (r.computed ? " · checked " + r.computed : "");
      box.appendChild(head);
      const t = document.createElement("table");
      t.innerHTML = "<thead><tr><th>bars in common</th><th>close identical</th><th>within a tick</th><th>max diff</th><th>volume identical</th><th>RTH sessions</th><th>count mismatches</th><th>days differing</th></tr></thead>";
      const tb = document.createElement("tbody");
      const cells = [
        (r.common_rows || 0).toLocaleString(), (c.identical_pct ?? "–") + "%", (c.within_tick_pct ?? "–") + "%",
        c.max_abs_diff ?? "–", (r.volume_identical_pct ?? "–") + "%", r.rth_sessions_compared ?? "–",
        (r.rth_count_mismatches || []).length, days.length];
      tb.innerHTML = "<tr>" + cells.map(v => "<td>" + v + "</td>").join("") + "</tr>";
      t.appendChild(tb);
      box.appendChild(t);
      if (days.length) {
        const note = document.createElement("div");
        note.className = "note";
        note.style.marginTop = "6px";
        note.textContent = "Days differing by more than a tick (a hand-rolled contract against a volume roll shows the calendar spread): " +
          days.map(d => d.date + " (" + (d.mean_diff >= 0 ? "+" : "") + d.mean_diff + ")").join(", ");
        box.appendChild(note);
      }
      el.appendChild(box);
    }
  }

  $("runs").addEventListener("change", (e) => selectRun(e.target.value));
  loadData();
  for (const b of ["standard", "comparable"]) {
    $("b-" + b).addEventListener("click", () => {
      state.basis = b;
      $("b-standard").classList.toggle("on", b === "standard");
      $("b-comparable").classList.toggle("on", b === "comparable");
      loadTrades();
    });
  }

  refreshRuns();
  setInterval(refreshRuns, 5000);
  setInterval(poll, 1000);
  setInterval(() => { if (!state.saved && state.events.length && state.events[state.events.length - 1].event === "stage") render(); }, 1000);
})();
</script>
</body>
</html>
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args(argv)
    import uvicorn  # noqa: PLC0415

    print(f"live backtest view: http://{args.host}:{args.port}  "
          f"(reading {live.LIVE_DIR})", flush=True)
    uvicorn.run(build_app(), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
