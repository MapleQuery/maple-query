/* ============================================================================
   MapleQuery ATLAS — atlas.js
   ----------------------------------------------------------------------------
   A command-driven UI. There is no sidebar and no tab bar. The whole product
   is three surfaces:

     STAGE  — a chrome-less writing document (prose + inline cited figures)
     ORBIT  — one floating command bar that asks, queries, navigates
     LENS   — a right slide-over data inspector

   Talks ONLY to MQ.api (data) and MQ.store (state). Backend logic is the
   untouched service layer copied from the previous build.
   ========================================================================== */
(function (global) {
  "use strict";
  var api = global.MQ.api, store = global.MQ.store;
  var reduce = global.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---- utils --------------------------------------------------------------- */
  function $(s, r) { return (r || document).querySelector(s); }
  function $$(s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); }
  function esc(s) { return String(s == null ? "" : s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  function el(html) { var t = document.createElement("template"); t.innerHTML = html.trim(); return t.content.firstElementChild; }
  function fmtDate(iso) { try { return new Date(iso).toISOString().slice(0, 10); } catch (e) { return iso; } }
  function uid(p) { return (p || "n") + "_" + Math.random().toString(36).slice(2, 8); }

  /* ---- icons --------------------------------------------------------------- */
  var FILLED = { spark: 1, pin: 1 };
  var ICON = {
    ask: '<circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/>',
    query: '<path d="M4 7h16M4 12h10M4 17h7"/><path d="m15 15 3 3 4-5" opacity=".0"/>',
    data: '<path d="M3 5h18v14H3z"/><path d="M3 10h18M9 5v14"/>',
    report: '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><path d="M14 2v6h6M9 13h6M9 17h4"/>',
    sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M4.9 4.9l1.4 1.4m11.4 11.4 1.4 1.4M2 12h2m16 0h2M4.9 19.1l1.4-1.4m11.4-11.4 1.4-1.4"/>',
    moon: '<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z"/>',
    plus: '<path d="M12 5v14M5 12h14"/>',
    check: '<path d="M20 6 9 17l-5-5"/>',
    spark: '<path d="M12 3l1.7 4.8L18.5 9.5l-4.8 1.7L12 16l-1.7-4.8L5.5 9.5l4.8-1.7L12 3Z"/>',
    arrow: '<path d="M5 12h14M13 6l6 6-6 6"/>',
    swap: '<path d="M7 4v16M7 20l-3-3M7 4l3 3M17 20V4M17 4l3 3M17 20l-3-3" opacity=".0"/><path d="M4 8h16M4 8l4-4M4 8l4 4M20 16H4M20 16l-4-4M20 16l-4 4"/>',
    warn: '<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/><path d="M12 9v4m0 4h.01"/>',
    pin: '<path d="M12 2 8 8H4l6 6-2 8 4-5 4 5-2-8 6-6h-4Z" opacity=".0"/><path d="M9 4h6l-1 6 4 3H6l4-3-1-6Z"/><path d="M12 13v7"/>',
    spin: '<path d="M21 12a9 9 0 1 1-6.2-8.6"/>',
    rerun: '<path d="M3 12a9 9 0 1 0 3-6.7"/><path d="M3 4v4h4"/>',
    doc: '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><path d="M14 2v6h6"/>',
    close: '<path d="M18 6 6 18M6 6l12 12"/>'
  };
  function svg(n, cls, sw) {
    var f = FILLED[n] ? 'fill="currentColor"' : 'fill="none" stroke="currentColor" stroke-width="' + (sw || 2) + '" stroke-linecap="round" stroke-linejoin="round"';
    return '<svg viewBox="0 0 24 24" class="' + (cls || "h-4 w-4") + '" ' + f + ' aria-hidden="true">' + ICON[n] + "</svg>";
  }

  /* ---- toast --------------------------------------------------------------- */
  var toastEl = $("#toast"), toastT;
  function toast(m) {
    toastEl.textContent = m;
    toastEl.classList.add("opacity-100", "translate-y-0");
    toastEl.classList.remove("opacity-0", "translate-y-3");
    clearTimeout(toastT);
    toastT = setTimeout(function () { toastEl.classList.remove("opacity-100", "translate-y-0"); toastEl.classList.add("opacity-0", "translate-y-3"); }, 2300);
  }

  /* ---- theme --------------------------------------------------------------- */
  function initTheme() {
    var saved = null; try { saved = localStorage.getItem("mq-atlas-theme"); } catch (e) {}
    applyTheme(saved || (global.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
    $("#themeToggle").addEventListener("click", function () { applyTheme(store.get().theme === "dark" ? "light" : "dark"); });
  }
  function applyTheme(t) {
    document.documentElement.setAttribute("data-theme", t);
    store.setTheme(t);
    try { localStorage.setItem("mq-atlas-theme", t); } catch (e) {}
    $("#themeIcon").innerHTML = svg(t === "dark" ? "sun" : "moon", "h-4 w-4");
  }

  /* ---- dataset cache + citation registry ---------------------------------- */
  var DATASETS = {};   // id -> full dataset (for the Lens)
  var SRC2DS = {};     // provenance.sourceId -> dataset id
  var citeReg = {};    // sourceId -> { n, prov }
  var citeSeq = 0;
  function citeNumber(prov) {
    if (!prov) return null;
    if (!citeReg[prov.sourceId]) citeReg[prov.sourceId] = { n: ++citeSeq, prov: prov };
    return citeReg[prov.sourceId].n;
  }
  function countSources() {
    var ids = {};
    store.get().blocks.forEach(function (b) {
      var p = b.provenance || (b.result && b.result.provenance);
      if (p) ids[p.sourceId] = 1;
    });
    return Math.max(1, Object.keys(ids).length);
  }

  /* Living citation: inline marker + hover peek card. */
  function citeMark(prov) {
    var n = citeNumber(prov);
    return '<span class="cite relative inline-block align-baseline" tabindex="0">' +
      '<span class="cite-mark" data-src="' + esc(prov.sourceId) + '">[' + n + ']</span>' +
      '<span class="peek absolute bottom-full left-1/2 z-20 mb-2 w-64 -translate-x-1/2 rounded-xl border border-line bg-raised p-3 text-left shadow-pop">' +
        '<span class="block font-mono text-[10px] uppercase tracking-[0.14em] text-accent">Source [' + n + ']</span>' +
        '<span class="mt-1 block text-[13px] font-semibold leading-snug text-ink">' + esc(prov.title) + '</span>' +
        '<span class="mt-1 block font-mono text-[10px] text-muted">' + esc(prov.publisher) + ' · ' + esc(prov.coverage) + '</span>' +
        '<span class="mt-1.5 block font-mono text-[10px] text-faint">updated ' + esc(fmtDate(prov.sourceWatermark)) + ' · ' + prov.rowCount.toLocaleString() + ' rows · click to inspect</span>' +
      '</span></span>';
  }

  /* =========================================================================
   * STAGE
   * ======================================================================= */
  var blocksEl = $("#blocks");

  function docMeta() {
    var n = store.get().blocks.length, s = countSources();
    $("#docMeta").innerHTML =
      '<span>' + n + " block" + (n === 1 ? "" : "s") + "</span><span class=\"text-line-strong\">/</span>" +
      "<span>" + s + " source" + (s === 1 ? "" : "s") + "</span><span class=\"text-line-strong\">/</span>" +
      "<span>edited just now</span>";
  }

  function grip() {
    return '<button class="grip absolute -left-6 top-1.5 rounded p-0.5 text-faint hover:text-muted" aria-label="Drag to reorder" tabindex="-1"><svg viewBox="0 0 24 24" class="h-4 w-4" fill="currentColor"><circle cx="9" cy="6" r="1.4"/><circle cx="15" cy="6" r="1.4"/><circle cx="9" cy="12" r="1.4"/><circle cx="15" cy="12" r="1.4"/><circle cx="9" cy="18" r="1.4"/><circle cx="15" cy="18" r="1.4"/></svg></button>';
  }
  function frame(inner, cls) {
    var w = el('<div class="blk settle group relative pl-1 ' + (cls || "") + '" draggable="true"></div>');
    w.innerHTML = grip() + inner;
    return w;
  }

  function proseBlock(b) {
    var w = frame('<p contenteditable="true" data-ph="Write, or press / for a query…" class="rounded px-0.5 text-[16.5px] leading-[1.75] text-ink-soft focus:outline-none">' + esc(b.text || "") + "</p>");
    w.dataset.id = b.id;
    var p = w.querySelector("p");
    p.addEventListener("input", function (e) { closeSlash(); store.patchBlock(b.id, { text: e.target.textContent }); });
    // Notion-style slash menu: "/" on an empty line offers query / ask / source.
    p.addEventListener("keydown", function (e) {
      if (e.key === "/" && !document.body.classList.contains("orbit-open") && p.textContent.trim() === "") {
        e.preventDefault();
        openSlashMenu(w, p, b);
      } else if (e.key === "Backspace" && slashMenu) {
        closeSlash();
      }
    });
    return w;
  }

  function sourceBlock(b) {
    var p = b.provenance || {};
    var w = frame('<div class="flex flex-wrap items-center gap-2 rounded-xl border border-line bg-canvas px-3 py-2 text-[13px]">' + svg("check", "h-4 w-4 text-ok") + '<span class="font-medium text-ink-soft">Pinned source</span>' + citeMark(p) + '<a href="' + esc(p.url || "#") + '" class="font-medium text-ink hover:text-accent">' + esc(p.title) + "</a></div>");
    w.dataset.id = b.id;
    return w;
  }

  function answerBlock(b) {
    var mark = b.provenance ? " " + citeMark(b.provenance) : "";
    var w = frame(
      '<div class="relative rounded-xl border-l-2 border-accent/40 bg-canvas/60 py-1 pl-4 pr-1">' +
        '<div class="mb-1 flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-[0.14em] text-accent">' + svg("spark", "h-3 w-3") + " Answer</div>" +
        '<p class="text-[16px] leading-[1.7] text-ink-soft">' + (b.html || "") + mark + "</p>" +
      "</div>");
    w.dataset.id = b.id;
    return w;
  }

  function queryBlock(b) {
    var w = frame(
      '<div class="rounded-xl border border-line bg-canvas/50 p-1">' +
        '<div class="flex items-center gap-2 px-2 py-1.5">' +
          '<span class="inline-flex items-center gap-1 font-mono text-[10px] font-semibold uppercase tracking-[0.14em] text-accent">' + svg("spark", "h-3 w-3") + " Query</span>" +
          '<input class="nl min-w-0 flex-1 bg-transparent font-mono text-[12.5px] text-ink placeholder:text-faint focus:outline-none" value="' + esc(b.nl || "") + '" placeholder="Describe the figure you want…" aria-label="Query prompt" />' +
          '<span class="stale hidden shrink-0 items-center gap-1 rounded-full bg-warn/15 px-2 py-0.5 font-mono text-[10px] font-medium text-warn"><span class="stxt">re-run</span></span>' +
          '<button class="run inline-flex shrink-0 cursor-pointer items-center gap-1 rounded-lg bg-accent px-2.5 py-1 font-mono text-[11px] font-semibold text-white transition-colors hover:bg-accent-press">' + svg("arrow", "h-3 w-3") + " Run</button>" +
        "</div>" +
        '<div class="sqlwrap hidden px-2"></div>' +
        '<div class="res hidden px-1 pb-1"></div>' +
      "</div>");
    w.dataset.id = b.id;
    var run = w.querySelector(".run"), nl = w.querySelector(".nl");
    run.addEventListener("click", function () { runQuery(w, b); });
    nl.addEventListener("input", function () { store.patchBlock(b.id, { nl: nl.value }); if (b.result && b.result.status === "ran") markStale(w, b); });
    nl.addEventListener("keydown", function (e) { if ((e.metaKey || e.ctrlKey) && e.key === "Enter") { e.preventDefault(); runQuery(w, b); } });
    if (b.sql) renderSql(w, b.sql, true);
    if (b.result) { w.querySelector(".res").classList.remove("hidden"); renderResult(w.querySelector(".res"), b.result); ran(run); }
    return w;
  }

  function ran(btn) { btn.innerHTML = svg("check", "h-3 w-3") + " Ran"; btn.classList.remove("bg-accent", "hover:bg-accent-press"); btn.classList.add("bg-ok", "hover:bg-ok"); }
  function resetRun(btn, label, icon) { btn.innerHTML = svg(icon || "arrow", "h-3 w-3") + " " + (label || "Run"); btn.classList.add("bg-accent", "hover:bg-accent-press"); btn.classList.remove("bg-ok", "hover:bg-ok"); btn.disabled = false; }
  function markStale(w, b) { b.result.status = "stale"; var p = w.querySelector(".stale"); p.classList.remove("hidden"); p.classList.add("inline-flex"); p.querySelector(".stxt").textContent = "prompt changed · re-run"; resetRun(w.querySelector(".run"), "Re-run", "rerun"); }

  function renderSql(w, sql, collapsed) {
    var wrap = w.querySelector(".sqlwrap"); wrap.classList.remove("hidden");
    wrap.innerHTML =
      '<button class="tog flex w-full items-center gap-2 py-1 text-left font-mono text-[10px] font-medium uppercase tracking-wide text-muted hover:text-ink">' +
        '<span class="cv transition-transform">' + '<svg viewBox="0 0 24 24" class="h-3.5 w-3.5" fill="none" stroke="currentColor" stroke-width="2"><path d="m6 9 6 6 6-6"/></svg>' + "</span>" +
        '<span class="text-accent">' + svg("spark", "h-2.5 w-2.5") + "</span> AI-compiled SQL <span class=\"ml-auto normal-case text-faint\">review before trusting</span></button>" +
      '<pre class="body scroll overflow-x-auto rounded-lg border border-line bg-paper px-3 py-2.5 font-mono text-[11.5px] leading-relaxed text-ink-soft ' + (collapsed ? "hidden" : "") + '"><code>' + esc(sql) + "</code></pre>";
    var body = wrap.querySelector(".body"), cv = wrap.querySelector(".cv");
    if (collapsed) cv.style.transform = "rotate(-90deg)";
    wrap.querySelector(".tog").addEventListener("click", function () { var h = body.classList.toggle("hidden"); cv.style.transform = h ? "rotate(-90deg)" : ""; });
  }

  function runQuery(w, b) {
    var run = w.querySelector(".run"), nl = w.querySelector(".nl").value.trim();
    if (!nl) { w.querySelector(".nl").focus(); return; }
    w.querySelector(".stale").classList.add("hidden");
    run.disabled = true; run.innerHTML = svg("spin", "h-3 w-3 animate-spin") + " Compiling";
    var ctx = { investigationId: store.get().investigation.id, blockId: b.id, nl: nl };
    api.compileQuery(ctx).then(function (plan) {
      store.patchBlock(b.id, { sql: plan.sql });
      renderSql(w, plan.sql, false);
      run.innerHTML = svg("spin", "h-3 w-3 animate-spin") + " Running";
      var res = w.querySelector(".res"); res.classList.remove("hidden");
      res.innerHTML = '<div class="space-y-2 p-3"><div class="sk h-3 w-2/3 rounded"></div><div class="sk h-28 w-full rounded"></div></div>';
      return api.runQuery(ctx);
    }).then(function (r) {
      store.patchBlock(b.id, { result: r });
      var res = w.querySelector(".res"); renderResult(res, r);
      res.classList.add("flash"); setTimeout(function () { res.classList.remove("flash"); }, 1100);
      ran(run); run.disabled = false; docMeta(); refreshStatus();
      toast("Query settled · figure + citation inline");
    }).catch(function (err) {
      resetRun(run, "Retry", "rerun");
      w.querySelector(".res").innerHTML = '<div class="flex items-start gap-2 p-3 text-sm text-warn">' + svg("warn", "mt-0.5 h-4 w-4") + '<span class="text-ink-soft">' + esc(err.message) + "</span></div>";
    });
  }

  function fmt(v, t) {
    if (t === "money") return "$" + Number(v).toFixed(2) + "B";
    if (t === "pct") return (v > 0 ? "+" : "") + v + "%";
    if (t === "num") return "$" + Number(v).toLocaleString();
    if (t === "int") return Number(v).toLocaleString();
    if (t === "big") return Number(v).toFixed(1) + "M";
    return esc(v);
  }
  function countUp(elm) {
    var to = parseFloat(elm.getAttribute("data-to")), dec = parseInt(elm.getAttribute("data-dec") || "0", 10);
    if (reduce) { elm.textContent = to.toFixed(dec); return; }
    var start = null;
    function step(ts) { if (start === null) start = ts; var p = Math.min((ts - start) / 900, 1); elm.textContent = (to * (1 - Math.pow(1 - p, 3))).toFixed(dec); if (p < 1) requestAnimationFrame(step); else elm.textContent = to.toFixed(dec); }
    requestAnimationFrame(step);
  }

  function renderResult(container, r) {
    var max = Math.max.apply(null, r.series.map(function (d) { return d.value; }));
    var bars = r.series.map(function (d, i) {
      var isMax = d.value === max;
      var a = 0.38 + 0.52 * (i / Math.max(1, r.series.length - 1));
      var bg = isMax ? "rgb(var(--accent))" : "rgb(var(--graph-2) / " + a.toFixed(2) + ")";
      return '<div class="flex flex-1 flex-col items-center gap-1.5"><div class="flex h-32 w-full items-end"><div class="col w-full rounded-t-sm" style="height:0%;background:' + bg + '"></div></div>' +
        '<span class="font-mono text-[10px] ' + (isMax ? "font-semibold text-accent" : "text-muted") + '">' + d.value + "</span>" +
        '<span class="font-mono text-[10px] text-faint">' + esc(d.label) + "</span></div>";
    }).join("");

    var cols = r.table.columns, rows = r.table.rows;
    var thead = '<tr class="border-b border-line font-mono text-[10px] uppercase tracking-wide text-faint">' + cols.map(function (c) { return '<th class="px-2.5 py-1.5 font-medium ' + (c.type !== "text" ? "text-right" : "text-left") + '">' + esc(c.label) + "</th>"; }).join("") + "</tr>";
    var tbody = rows.map(function (row) { return '<tr class="border-b border-line/60 last:border-0">' + cols.map(function (c, ci) { var col = c.type === "pct" ? (row[c.key] >= 0 ? "text-ok" : "text-warn") : ""; return '<td class="px-2.5 py-1.5 ' + (c.type !== "text" ? "text-right font-mono" : "") + " " + (ci === 0 ? "font-medium text-ink" : "") + " " + col + '">' + fmt(row[c.key], c.type) + "</td>"; }).join("") + "</tr>"; }).join("");

    var gaps = (r.gaps || []).map(function (g) { return '<div class="mt-2 flex items-start gap-1.5 rounded-lg bg-warn/10 px-2.5 py-1.5 text-[11px]">' + svg("warn", "mt-0.5 h-3.5 w-3.5 text-warn") + '<span class="text-ink-soft"><span class="font-medium text-ink">Honest gap · ' + esc(g.dim) + ":</span> " + esc(g.note) + "</span></div>"; }).join("");

    var p = r.provenance, n = citeNumber(p);
    container.innerHTML =
      '<div class="p-2">' +
        '<div class="mb-3 flex items-center justify-between gap-2 px-1">' +
          '<h3 class="font-display text-[13px] font-semibold text-ink">' + esc(r.title) + "</h3>" +
          '<div class="flex gap-0.5 rounded-lg bg-canvas p-0.5">' +
            '<button class="vt rounded-md px-2 py-0.5 font-mono text-[10px] font-semibold text-ink" data-v="bar">bar</button>' +
            '<button class="vt rounded-md px-2 py-0.5 font-mono text-[10px] font-medium text-muted" data-v="table">table</button>' +
          "</div>" +
        "</div>" +
        '<div class="viewbar flex h-44 items-end gap-2 px-1">' + bars + "</div>" +
        '<div class="viewtab hidden overflow-x-auto"><table class="w-full text-left text-[12.5px]"><thead>' + thead + "</thead><tbody class=\"text-ink\">" + tbody + "</tbody></table></div>" +
        (r.delta ? '<p class="mt-3 px-1 text-[13px] text-ink-soft">Headline <span class="rounded bg-accent/10 px-1 font-semibold text-accent">' + esc(r.delta.prefix || "") + '<span class="cu" data-to="' + r.delta.value + '" data-dec="' + (r.delta.dec || 0) + '">0</span>' + esc(r.delta.suffix || "") + "</span> " + esc(r.delta.label) + "</p>" : "") +
        gaps +
        '<div class="mt-3 flex flex-wrap items-center gap-x-2 gap-y-1 border-t border-line px-1 pt-2 text-[11px]">' + svg("check", "h-3.5 w-3.5 text-ok") + citeMark(p) +
          '<a href="' + esc(p.url) + '" class="font-medium text-ink hover:text-accent">' + esc(p.title) + "</a>" +
          '<span class="font-mono text-[10px] text-muted">' + esc(p.publisher) + " · updated " + esc(fmtDate(p.sourceWatermark)) + " · " + p.rowCount.toLocaleString() + " rows</span></div>" +
      "</div>";

    container.querySelectorAll(".col").forEach(function (bar, i) { var pct = (r.series[i].value / max) * 100; if (reduce) bar.style.height = pct + "%"; else setTimeout(function () { bar.style.height = pct + "%"; }, 70 * i); });
    container.querySelectorAll(".cu").forEach(countUp);
    container.querySelectorAll(".vt").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var v = btn.getAttribute("data-v");
        container.querySelectorAll(".vt").forEach(function (x) { var on = x === btn; x.classList.toggle("text-ink", on); x.classList.toggle("font-semibold", on); x.classList.toggle("text-muted", !on); x.classList.toggle("font-medium", !on); });
        container.querySelector(".viewbar").classList.toggle("hidden", v !== "bar");
        container.querySelector(".viewtab").classList.toggle("hidden", v !== "table");
      });
    });
  }

  /* ---- block rendering + insertion ---------------------------------------- */
  function renderBlock(b) {
    if (b.type === "query") return queryBlock(b);
    if (b.type === "answer") return answerBlock(b);
    if (b.type === "source") return sourceBlock(b);
    return proseBlock(b);
  }
  function renderBlocks() { blocksEl.innerHTML = ""; store.get().blocks.forEach(function (b) { blocksEl.appendChild(renderBlock(b)); }); docMeta(); }

  function appendBlock(seed, opts) {
    opts = opts || {};
    return api.createBlock({ investigationId: store.get().investigation.id, block: seed }).then(function (saved) {
      var atIndex, afterNode = null;
      if (opts.afterId) {
        var idx = store.get().blocks.findIndex(function (x) { return x.id === opts.afterId; });
        if (idx > -1) { atIndex = idx + 1; afterNode = blocksEl.querySelector('[data-id="' + opts.afterId + '"]'); }
      }
      var block = store.addBlock(saved, atIndex);
      var node = renderBlock(block);
      if (afterNode) afterNode.after(node); else blocksEl.appendChild(node);
      docMeta(); refreshStatus();
      if (opts.scroll !== false) node.scrollIntoView({ block: "center", behavior: reduce ? "auto" : "smooth" });
      var f = node.querySelector("[contenteditable], .nl"); if (opts.focus && f) f.focus();
      return { block: block, node: node };
    });
  }

  function addProse() { appendBlock({ type: "prose", text: "" }, { focus: true }); }

  function insertQuery(nl) {
    return appendBlock({ type: "query", nl: nl || "New figure — describe it in plain language", sql: null, result: null }).then(function (h) {
      if (nl) runQuery(h.node, h.block); else { var i = h.node.querySelector(".nl"); if (i) i.focus(); }
    });
  }

  function insertAnswer(question) {
    // Ask flows straight into the document as an inline, cited answer.
    return appendBlock({ type: "answer", html: '<span class="dots inline-flex items-center gap-1 align-middle"><i></i><i></i><i></i></span>', provenance: null }, { scroll: true }).then(function (h) {
      return api.askAgent(question).then(function (resp) {
        var html = resp.answer
          ? resp.answer
          : '<span class="text-ink">Not enough evidence for that yet.</span> Try IT contract spend, housing grants by province, or the immigration backlog.';
        store.patchBlock(h.block.id, { html: html, provenance: resp.citation || null, q: question });
        // re-render just this node
        var fresh = renderBlock(store.get().blocks.find(function (x) { return x.id === h.block.id; }));
        h.node.replaceWith(fresh);
        docMeta(); refreshStatus();
      });
    });
  }

  /* ---- slash menu: "/" on an empty line, Notion-style ---------------------- */
  var slashMenu = null, slashHost = null;
  function slashOpts() { return $$(".so", slashMenu); }
  function slashIndex() { var os = slashOpts(); for (var i = 0; i < os.length; i++) if (os[i].getAttribute("aria-selected") === "true") return i; return 0; }
  function slashPaint(i) { slashOpts().forEach(function (o, k) { var on = k === i; o.setAttribute("aria-selected", on ? "true" : "false"); o.classList.toggle("bg-canvas", on); }); }
  function slashOutside(e) { if (slashMenu && !slashMenu.contains(e.target)) closeSlash(); }
  function slashKeys(e) {
    if (!slashMenu) return;
    var os = slashOpts(), cur = slashIndex();
    if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); var h = slashHost; closeSlash(); if (h) h.focus(); }
    else if (e.key === "ArrowDown") { e.preventDefault(); e.stopPropagation(); slashPaint(Math.min(os.length - 1, cur + 1)); }
    else if (e.key === "ArrowUp") { e.preventDefault(); e.stopPropagation(); slashPaint(Math.max(0, cur - 1)); }
    else if (e.key === "Enter" || e.key === "Tab") { e.preventDefault(); e.stopPropagation(); if (os[cur]) os[cur].click(); }
  }
  function closeSlash() {
    if (!slashMenu) return;
    slashMenu.remove(); slashMenu = null; slashHost = null;
    document.removeEventListener("mousedown", slashOutside, true);
    document.removeEventListener("keydown", slashKeys, true);
  }
  function openSlashMenu(wrap, host, block) {
    closeSlash();
    var actions = [
      { icon: "query", label: "Insert a live query", hint: "figure", run: function () { convertToQuery(wrap, block); } },
      { icon: "ask",   label: "Ask a question",       hint: "answer", run: function () { openOrbit(); } },
      { icon: "data",  label: "Cite a source",        hint: "lens",   run: function () { openLens(); } },
      { icon: "plus",  label: "Keep as text",         hint: "",       run: function () { host.focus(); } }
    ];
    var menu = el('<div class="slash-menu absolute left-1 top-8 z-30 w-64 overflow-hidden rounded-xl border border-line bg-raised py-1 shadow-pop"></div>');
    menu.innerHTML =
      '<div class="px-3 pb-1 pt-1.5 font-mono text-[10px] uppercase tracking-[0.14em] text-faint">Insert</div>' +
      actions.map(function (it, i) {
        return '<button data-i="' + i + '" aria-selected="false" class="so flex w-full cursor-pointer items-center gap-3 px-3 py-2 text-left">' +
          '<span class="grid h-7 w-7 shrink-0 place-items-center rounded-lg bg-canvas text-muted">' + svg(it.icon, "h-4 w-4") + '</span>' +
          '<span class="flex-1 truncate text-[14px] text-ink-soft">' + esc(it.label) + '</span>' +
          (it.hint ? '<span class="cap">' + esc(it.hint) + '</span>' : '') +
          '</button>';
      }).join("");
    wrap.appendChild(menu);
    slashMenu = menu; slashHost = host;
    slashPaint(0);
    slashOpts().forEach(function (btn, i) {
      btn.addEventListener("mousemove", function () { slashPaint(i); });
      btn.addEventListener("mousedown", function (e) { e.preventDefault(); });   // keep the caret put until click resolves
      btn.addEventListener("click", function () { closeSlash(); actions[i].run(); });
    });
    setTimeout(function () {
      document.addEventListener("mousedown", slashOutside, true);
      document.addEventListener("keydown", slashKeys, true);
    }, 0);
  }
  // Turn the empty launcher block into a live query block in place.
  function convertToQuery(wrap, block) {
    appendBlock({ type: "query", nl: "", sql: null, result: null }, { afterId: block.id, focus: true }).then(function (h) {
      if ((block.text || "").trim() === "") {
        store.removeBlock(block.id);
        if (wrap && wrap.parentNode) wrap.remove();
        docMeta();
      }
      var i = h.node.querySelector(".nl"); if (i) i.focus();
    });
  }

  /* ---- drag reorder -------------------------------------------------------- */
  var dragEl = null;
  blocksEl.addEventListener("dragstart", function (e) { var b = e.target.closest(".blk"); if (!b) return; if (e.target.closest("input,[contenteditable],button.run,.vt")) { e.preventDefault(); return; } dragEl = b; b.classList.add("drag"); e.dataTransfer.effectAllowed = "move"; });
  blocksEl.addEventListener("dragend", function () { if (dragEl) dragEl.classList.remove("drag"); $$(".over-top,.over-bot", blocksEl).forEach(function (x) { x.classList.remove("over-top", "over-bot"); }); dragEl = null; });
  blocksEl.addEventListener("dragover", function (e) { e.preventDefault(); var o = e.target.closest(".blk"); $$(".over-top,.over-bot", blocksEl).forEach(function (x) { x.classList.remove("over-top", "over-bot"); }); if (o && o !== dragEl) { var r = o.getBoundingClientRect(); o.classList.add(e.clientY > r.top + r.height / 2 ? "over-bot" : "over-top"); } });
  blocksEl.addEventListener("drop", function (e) { e.preventDefault(); var o = e.target.closest(".blk"); if (!o || !dragEl || o === dragEl) return; var r = o.getBoundingClientRect(); var after = e.clientY > r.top + r.height / 2; blocksEl.insertBefore(dragEl, after ? o.nextSibling : o); var nb = dragEl.nextElementSibling; store.moveBlock(dragEl.dataset.id, nb ? nb.dataset.id : null); api.reorderBlocks({ investigationId: store.get().investigation.id, order: store.get().blocks.map(function (b) { return b.id; }) }); toast("Reordered · citations stable"); });

  /* ---- citation clicks (delegated) → open Lens ---------------------------- */
  blocksEl.addEventListener("click", function (e) {
    var m = e.target.closest(".cite-mark"); if (!m) return;
    var dsId = SRC2DS[m.getAttribute("data-src")];
    if (dsId) openLens(dsId);
  });

  /* =========================================================================
   * ORBIT — the one command surface
   * ======================================================================= */
  var orbit = $("#orbit"), pill = $("#orbitPill"), panel = $("#orbitPanel"), oInput = $("#orbitInput"), oMode = $("#orbitMode"), oSuggest = $("#orbitSuggest"), oForm = $("#orbitForm");
  var selIdx = -1, curItems = [];
  var recents = [];

  function parseMode(v) {
    v = v || "";
    if (/^\/(q|query)\b/i.test(v)) return { mode: "query", arg: v.replace(/^\/(q|query)\s*/i, "") };
    if (/^\/(d|data|lens)\b/i.test(v)) return { mode: "data", arg: v.replace(/^\/(d|data|lens)\s*/i, "") };
    if (/^\/(r|report)\b/i.test(v) || /^>\s*promote/i.test(v)) return { mode: "report", arg: "" };
    if (/^>/.test(v)) return { mode: "command", arg: v.replace(/^>\s*/, "") };
    return { mode: "ask", arg: v };
  }
  var MODE_LABEL = { ask: "Ask", query: "Query", data: "Lens", report: "Report", command: "Cmd" };

  function commands() {
    return [
      { icon: "query", label: "Insert a live query", hint: "/query", run: function () { openOrbitWith("/query "); } },
      { icon: "data", label: "Open the Data Lens", hint: "/data", run: function () { openLens(); } },
      { icon: "report", label: "Compile to report", hint: "promote", run: openModal },
      { icon: "doc", label: "Switch document…", hint: "⌘\\", run: openLibrary },
      { icon: "doc", label: "New investigation", hint: "", run: newDoc },
      { icon: (store.get().theme === "dark" ? "sun" : "moon"), label: "Toggle theme", hint: "", run: function () { applyTheme(store.get().theme === "dark" ? "light" : "dark"); } },
      { icon: "plus", label: "New text block", hint: "", run: addProse }
    ];
  }

  function renderSuggest(v) {
    var pm = parseMode(v);
    oMode.textContent = MODE_LABEL[pm.mode];
    var items = [];

    if (pm.mode === "ask") {
      if (pm.arg.trim()) items.push({ icon: "spark", label: "Ask: " + pm.arg.trim(), hint: "enter", primary: true, run: function () { doAsk(pm.arg.trim()); } });
      var ex = ["How has federal IT contract spend changed since 2018?", "Which provinces got the most housing grants?", "What happened to the immigration backlog?"];
      (pm.arg.trim() ? recents.slice(0, 3) : ex).forEach(function (q) { items.push({ icon: "ask", label: q, hint: "", run: function () { doAsk(q); } }); });
      commands().slice(0, 3).forEach(function (c) { items.push(c); });
    } else if (pm.mode === "query") {
      items.push({ icon: "spark", label: pm.arg.trim() ? "Insert live query: " + pm.arg.trim() : "Describe a figure to insert…", hint: "enter", primary: true, run: function () { if (pm.arg.trim()) doQuery(pm.arg.trim()); } });
    } else if (pm.mode === "data") {
      Object.keys(DATASETS).forEach(function (id) { var d = DATASETS[id]; if (!pm.arg || (d.title + d.category).toLowerCase().indexOf(pm.arg.toLowerCase()) > -1) items.push({ icon: "data", label: d.title, hint: d.category, run: function () { openLens(id); } }); });
    } else if (pm.mode === "report") {
      items.push({ icon: "report", label: "Compile stage into a sourced report", hint: "enter", primary: true, run: openModal });
    } else {
      commands().forEach(function (c) { if (!pm.arg || c.label.toLowerCase().indexOf(pm.arg.toLowerCase()) > -1) items.push(c); });
    }

    curItems = items;
    if (selIdx >= items.length) selIdx = items.length - 1;
    oSuggest.innerHTML = items.length ? items.map(function (it, i) {
      return '<button data-i="' + i + '" class="sg flex w-full cursor-pointer items-center gap-3 px-4 py-2.5 text-left ' + (i === selIdx ? "bg-canvas" : "") + '">' +
        '<span class="grid h-7 w-7 shrink-0 place-items-center rounded-lg ' + (it.primary ? "bg-accent text-white" : "bg-canvas text-muted") + '">' + svg(it.icon, "h-4 w-4") + "</span>" +
        '<span class="flex-1 truncate text-[14px] ' + (it.primary ? "font-medium text-ink" : "text-ink-soft") + '">' + esc(it.label) + "</span>" +
        (it.hint ? '<span class="cap">' + esc(it.hint) + "</span>" : "") + "</button>";
    }).join("") : '<div class="px-4 py-6 text-center text-sm text-muted">Nothing here — press enter to ask.</div>';
    $$(".sg", oSuggest).forEach(function (btn) {
      btn.addEventListener("mousemove", function () { selIdx = +btn.getAttribute("data-i"); paintSel(); });
      btn.addEventListener("click", function () { var it = curItems[+btn.getAttribute("data-i")]; if (it) { closeOrbit(); it.run(); } });
    });
  }
  function paintSel() { $$(".sg", oSuggest).forEach(function (b, i) { b.classList.toggle("bg-canvas", i === selIdx); }); }

  function openOrbit() { document.body.classList.add("orbit-open"); panel.classList.remove("hidden"); selIdx = -1; renderSuggest(oInput.value); setTimeout(function () { oInput.focus(); }, 20); }
  function openOrbitWith(prefix) { openOrbit(); oInput.value = prefix; renderSuggest(oInput.value); oInput.focus(); }
  function closeOrbit() { document.body.classList.remove("orbit-open"); panel.classList.add("hidden"); oInput.value = ""; selIdx = -1; }

  pill.addEventListener("click", openOrbit);
  $("#workspaceMenu").addEventListener("click", function (e) { e.preventDefault(); openLibrary(); });
  oInput.addEventListener("input", function () { selIdx = -1; renderSuggest(this.value); });
  oInput.addEventListener("keydown", function (e) {
    if (e.key === "ArrowDown") { e.preventDefault(); selIdx = Math.min(curItems.length - 1, selIdx + 1); paintSel(); }
    else if (e.key === "ArrowUp") { e.preventDefault(); selIdx = Math.max(-1, selIdx - 1); paintSel(); }
  });
  oForm.addEventListener("submit", function (e) {
    e.preventDefault();
    if (selIdx >= 0 && curItems[selIdx]) { var it = curItems[selIdx]; closeOrbit(); it.run(); return; }
    var pm = parseMode(oInput.value);
    closeOrbit();
    if (pm.mode === "ask" && pm.arg.trim()) doAsk(pm.arg.trim());
    else if (pm.mode === "query" && pm.arg.trim()) doQuery(pm.arg.trim());
    else if (pm.mode === "data") openLens();
    else if (pm.mode === "report") openModal();
    else if (pm.mode === "command" && curItems[0]) curItems[0].run();
  });

  function doAsk(q) { recents.unshift(q); recents = recents.slice(0, 6); insertAnswer(q); }
  function doQuery(nl) { insertQuery(nl); }

  /* =========================================================================
   * LENS — right slide-over data inspector
   * ======================================================================= */
  var lens = $("#lens"), lensSheet = $("#lensSheet"), curDs = null, lensSearch = "", lensSort = null, lensDir = 1;

  function openLens(id) {
    lens.classList.remove("hidden");
    requestAnimationFrame(function () { document.body.classList.add("lens-open"); });
    renderChips();
    loadLens(id || (curDs && curDs.id) || Object.keys(DATASETS)[0]);
    setTimeout(function () { $("#lensClose").focus(); }, 60);
  }
  function closeLens() { document.body.classList.remove("lens-open"); setTimeout(function () { lens.classList.add("hidden"); }, 400); }
  function loadLens(id) { curDs = DATASETS[id]; lensSearch = ""; lensSort = null; lensDir = 1; $("#lensSearch").value = ""; paintLens(); }

  function renderChips() {
    $("#lensChips").innerHTML = Object.keys(DATASETS).map(function (id) {
      var d = DATASETS[id], on = curDs && curDs.id === id;
      return '<button data-id="' + id + '" class="chip cursor-pointer rounded-full border px-3 py-1 text-xs font-medium transition-colors ' + (on ? "border-accent bg-accent/10 text-accent" : "border-line text-muted hover:text-ink") + '">' + esc(d.title.split(" ").slice(0, 2).join(" ")) + "</button>";
    }).join("");
    $$("#lensChips .chip").forEach(function (b) { b.addEventListener("click", function () { loadLens(b.getAttribute("data-id")); }); });
  }
  function paintLens() {
    if (!curDs) return;
    renderChips();
    $("#lensTitle").textContent = curDs.title;
    $("#lensDesc").textContent = curDs.desc;
    $("#lensSource").innerHTML = svg("check", "mt-0.5 h-3.5 w-3.5 shrink-0 text-ok") + "<span>" + esc(curDs.provenance.title) + " · " + esc(curDs.provenance.publisher) + "</span>";
    var m = $("#lensMeta"); m.innerHTML = "";
    Object.keys(curDs.meta).forEach(function (k) { var amber = k === "Completeness" && parseInt(curDs.meta[k], 10) < 100; m.appendChild(el('<div class="bg-raised px-3 py-2.5"><dt class="font-mono text-[10px] uppercase tracking-wide text-faint">' + esc(k) + '</dt><dd class="mt-0.5 font-mono text-sm font-semibold ' + (amber ? "text-warn" : "text-ink") + '">' + esc(curDs.meta[k]) + "</dd></div>")); });
    lensTable();
  }
  function lensRows() {
    var rows = curDs.rows.slice();
    if (lensSearch) { var q = lensSearch.toLowerCase(); rows = rows.filter(function (r) { return curDs.columns.some(function (c) { return String(r[c.key]).toLowerCase().indexOf(q) > -1; }); }); }
    if (lensSort) rows.sort(function (a, b) { var x = a[lensSort], y = b[lensSort]; if (typeof x === "number" && typeof y === "number") return (x - y) * lensDir; return String(x).localeCompare(String(y)) * lensDir; });
    return rows;
  }
  function lensTable() {
    $("#lensHead").innerHTML = '<tr class="border-b border-line font-mono text-[10px] uppercase tracking-wide text-faint">' + curDs.columns.map(function (c) { var num = c.type !== "text", ar = lensSort === c.key ? (lensDir > 0 ? " ↑" : " ↓") : ""; return '<th class="px-3 py-2 font-medium ' + (num ? "text-right" : "") + '"><button class="hb cursor-pointer hover:text-ink" data-k="' + c.key + '">' + esc(c.label) + '<span class="text-accent">' + ar + "</span></button></th>"; }).join("") + "</tr>";
    $$("#lensHead .hb").forEach(function (b) { b.addEventListener("click", function () { var k = b.getAttribute("data-k"); if (lensSort === k) lensDir *= -1; else { lensSort = k; lensDir = 1; } lensTable(); }); });
    var rows = lensRows(), tb = $("#lensBody"); tb.innerHTML = "";
    rows.forEach(function (r) {
      var tr = el('<tr draggable="true" class="rin cursor-pointer border-b border-line/60 transition-colors last:border-0 hover:bg-canvas">' + curDs.columns.map(function (c, ci) { var num = c.type !== "text", col = c.type === "pct" ? (r[c.key] >= 0 ? "text-ok" : "text-warn") : ""; return '<td class="px-3 py-2 ' + (num ? "text-right font-mono" : "") + " " + (ci === 0 ? "font-medium text-ink" : "") + " " + col + '">' + fmt(r[c.key], c.type) + "</td>"; }).join("") + "</tr>");
      tr.addEventListener("click", function () { pinRow(r); });
      tr.addEventListener("dragstart", function (e) { e.dataTransfer.setData("text/mq-row", "1"); e.dataTransfer.effectAllowed = "copy"; pinPending = r; document.body.classList.add("row-dragging"); });
      tr.addEventListener("dragend", function () { document.body.classList.remove("row-dragging"); setTimeout(function () { pinPending = null; }, 0); });
      tb.appendChild(tr);
    });
    $("#lensEmpty").classList.toggle("hidden", rows.length > 0);
    $("#lensCount").textContent = rows.length + " of " + curDs.rows.length + " rows";
  }
  var pinPending = null;
  function pinRow(r) {
    var label = r[curDs.columns[0].key];
    appendBlock({ type: "source", provenance: curDs.provenance }).then(function () {
      toast("Pinned “" + label + "” · cited from " + curDs.title);
      closeLens();
    });
  }

  $("#lensClose").addEventListener("click", closeLens);
  $("#lensScrim").addEventListener("click", closeLens);
  $("#srcBtn").addEventListener("click", function () { openLens(); });
  $("#lensSearch").addEventListener("input", function () { var v = this.value; clearTimeout(window.__ls); window.__ls = setTimeout(function () { lensSearch = v; lensTable(); }, reduce ? 0 : 110); });
  $("#lensDownload").addEventListener("click", function () { toast("Downloading “" + curDs.title + "” as CSV…"); });
  // Drop a Lens row anywhere on the Stage to pin a cited figure.
  $("#stage").addEventListener("dragover", function (e) { if (pinPending) e.preventDefault(); });
  $("#stage").addEventListener("drop", function (e) { if (pinPending) { e.preventDefault(); var r = pinPending; pinPending = null; pinRow(r); } });

  /* =========================================================================
   * LIBRARY — left slide-over document navigator (Notion-style)
   * ======================================================================= */
  var library = $("#library"), libQuery = "";

  function libItem(inv) {
    var active = store.get().investigation && store.get().investigation.id === inv.id;
    return '<li><button data-id="' + inv.id + '" class="lib-item flex w-full cursor-pointer items-center gap-2 rounded-lg px-2 py-1.5 text-left transition-colors ' +
      (active ? "bg-accent/10 text-ink" : "text-ink-soft hover:bg-canvas") + '">' +
      '<span class="' + (active ? "text-accent" : "text-faint") + '">' + svg("doc", "h-4 w-4") + "</span>" +
      '<span class="min-w-0 flex-1 truncate ' + (active ? "font-medium" : "") + '">' + esc(inv.title) + "</span>" +
      '<span class="shrink-0 font-mono text-[10px] text-faint">' + esc(inv.updatedAt) + "</span></button></li>";
  }
  function renderLibrary() {
    var invs = store.get().investigations || [];
    var q = libQuery.toLowerCase();
    var match = function (i) { return !q || i.title.toLowerCase().indexOf(q) > -1; };
    var saved = invs.filter(function (i) { return i.saved && match(i); });
    var all = invs.filter(match);
    $("#libSaved").innerHTML = saved.length ? saved.map(libItem).join("") : '<li class="px-2 py-1 text-[12px] text-faint">No saved documents</li>';
    $("#libList").innerHTML = all.length ? all.map(libItem).join("") : '<li class="px-2 py-1 text-[12px] text-faint">Nothing found</li>';
    $$(".lib-item", library).forEach(function (b) { b.addEventListener("click", function () { switchDoc(b.getAttribute("data-id")); }); });
  }
  function openLibrary() { library.classList.remove("hidden"); requestAnimationFrame(function () { document.body.classList.add("lib-open"); }); renderLibrary(); setTimeout(function () { $("#libSearch").focus(); }, 60); }
  function closeLibrary() { document.body.classList.remove("lib-open"); setTimeout(function () { library.classList.add("hidden"); }, 400); }

  function loadDoc(inv, opts) {
    store.hydrate({ investigation: inv });
    citeReg = {}; citeSeq = 0;                 // citation numbers restart per document
    $("#docTitle").value = inv.title;
    renderBlocks(); docMeta(); refreshStatus();
    var fq = store.get().blocks.find(function (b) { return b.type === "query" && b.nl && !b.result; });
    if (fq) { var n = $('[data-id="' + fq.id + '"]'); if (n) setTimeout(function () { runQuery(n, fq); }, reduce ? 0 : 300); }
    if (opts && opts.focusTitle) { var dt = $("#docTitle"); dt.focus(); dt.select(); }
    document.getElementById("stage").scrollTop = 0;
  }
  function switchDoc(id) {
    if (store.get().investigation && store.get().investigation.id === id) { closeLibrary(); return; }
    closeLibrary();
    api.getInvestigation(id).then(function (inv) { loadDoc(inv); toast("Opened “" + inv.title + "”"); });
  }
  function newDoc() {
    var id = uid("inv");
    var inv = { id: id, title: "Untitled investigation", updatedAt: "just now", blocks: [] };
    store.get().investigations.unshift({ id: id, title: inv.title, updatedAt: "just now", saved: false });
    closeLibrary();
    loadDoc(inv, { focusTitle: true });
    toast("New investigation created");
  }

  $("#libClose").addEventListener("click", closeLibrary);
  $("#libScrim").addEventListener("click", closeLibrary);
  $("#libNew").addEventListener("click", newDoc);
  $("#libSearch").addEventListener("input", function () { libQuery = this.value; renderLibrary(); });

  /* =========================================================================
   * PROMOTE MODAL
   * ======================================================================= */
  var modal = $("#modal");
  function openModal() {
    api.promoteToReport({ investigationId: store.get().investigation.id, sourceCount: countSources(), blockCount: store.get().blocks.length }).then(function (rep) {
      $("#modalSrc").textContent = rep.sourceCount + " source" + (rep.sourceCount === 1 ? "" : "s");
      modal.classList.remove("hidden"); modal.classList.add("flex"); $("#modalClose").focus();
    });
  }
  function closeModal() { modal.classList.add("hidden"); modal.classList.remove("flex"); }
  $("#modalClose").addEventListener("click", closeModal);
  modal.addEventListener("click", function (e) { if (e.target === modal) closeModal(); });
  $$(".exp").forEach(function (b) { b.addEventListener("click", function () { closeModal(); toast("Exporting " + b.textContent.trim() + " · citations kept as footnotes"); }); });

  /* ---- status cluster ------------------------------------------------------ */
  function refreshStatus() { $("#srcCount").textContent = countSources(); }
  function renderStatus() {
    var s = store.get();
    var people = [s.session.user].concat(s.session.presence);
    $("#presence").innerHTML = people.map(function (p, i) {
      var onA = p.tint === "primary";
      return '<span class="grid h-7 w-7 place-items-center rounded-full border-2 border-canvas bg-' + (onA ? "accent" : "line-strong") + ' text-[10px] font-semibold ' + (onA ? "text-white" : "text-ink") + " " + (i ? "-ml-2" : "") + '" title="' + esc(p.name) + '">' + esc(p.initials) + "</span>";
    }).join("");
    $("#syncPill").classList.add("sm:inline-flex"); $("#syncPill").classList.remove("hidden");
    $("#srcBtn").classList.add("sm:inline-flex"); $("#srcBtn").classList.remove("hidden");
    refreshStatus();
  }

  /* =========================================================================
   * KEYBOARD
   * ======================================================================= */
  document.addEventListener("keydown", function (e) {
    var openK = (e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k";
    var altSpace = e.altKey && e.code === "Space";
    if (openK || altSpace) { e.preventDefault(); document.body.classList.contains("orbit-open") ? closeOrbit() : openOrbit(); return; }
    if ((e.metaKey || e.ctrlKey) && e.key === "\\") { e.preventDefault(); document.body.classList.contains("lib-open") ? closeLibrary() : openLibrary(); return; }
    if (e.key === "Escape") { if (!library.classList.contains("hidden")) closeLibrary(); if (!lens.classList.contains("hidden")) closeLens(); closeOrbit(); closeModal(); return; }
    if (e.key === "/" ) {
      var t = (e.target.tagName || "").toLowerCase();
      var editable = e.target.isContentEditable || t === "input" || t === "textarea";
      if (!editable && !document.body.classList.contains("orbit-open")) { e.preventDefault(); openOrbitWith("/query "); }
    }
  });
  $("#stageHint").addEventListener("click", addProse);

  /* =========================================================================
   * BOOT
   * ======================================================================= */
  function boot() {
    initTheme();
    Promise.all([api.getSession(), api.listSources(), api.listInvestigations(), api.getInvestigation("inv_it"), api.listDatasets()])
      .then(function (res) {
        store.hydrate({ session: res[0], sources: res[1], investigations: res[2], investigation: res[3] });
        $("#docTitle").value = res[3].title;
        // Preload full datasets for the Lens + citation routing.
        return Promise.all(res[4].map(function (d) { return api.getDataset(d.id); }));
      })
      .then(function (fulls) {
        fulls.forEach(function (d) { DATASETS[d.id] = d; SRC2DS[d.provenance.sourceId] = d.id; });
        renderStatus();
        renderBlocks();
        var fq = store.get().blocks.find(function (b) { return b.type === "query" && b.nl && !b.result; });
        if (fq) { var node = $('[data-id="' + fq.id + '"]'); if (node) setTimeout(function () { runQuery(node, fq); }, reduce ? 0 : 500); }
      })
      .catch(function (err) { toast("Boot failed: " + err.message); });

    $("#docTitle").addEventListener("input", function () {
      var t = this.value, inv = store.get().investigation; inv.title = t;
      var entry = (store.get().investigations || []).find(function (x) { return x.id === inv.id; });
      if (entry) entry.title = t;
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();

})(window);
