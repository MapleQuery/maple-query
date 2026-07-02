/* ============================================================================
   MapleQuery Notebook — api.js
   ----------------------------------------------------------------------------
   THE BACKEND SEAM. This is the only file in the app that knows where data
   comes from. Everything else (store.js, app.js) depends on these method
   signatures — never on fetch(), URLs, or mock arrays directly.

   To go live:
     1. Set MQ.config.mode = "live" (or ?mode=live in the URL).
     2. Point MQ.config.baseUrl at your API gateway.
     3. Implement the matching endpoints (see @endpoint tags on each method).
   The mock branch and the live branch return the SAME shapes, so no UI code
   changes when the backend arrives.

   Data shapes mirror the MapleQuery product spec (block + provenance model).
   ========================================================================== */
(function (global) {
  "use strict";

  var MQ = global.MQ || (global.MQ = {});

  /* -------------------------------------------------------------------------
   * Config — the switchboard
   * ---------------------------------------------------------------------- */
  var params = new URLSearchParams(global.location ? global.location.search : "");
  MQ.config = {
    mode: params.get("mode") || "mock",   // "mock" | "live"
    baseUrl: params.get("api") || "/api/v1",
    token: null,                            // set to a bearer token in live mode
    // Simulated network latency for the mock, so loading states are real.
    latency: { fast: 260, base: 620, query: 1050, agent: 900 }
  };

  /* =========================================================================
   * Type contracts (JSDoc) — the interface the UI is written against.
   * =========================================================================
   *
   * @typedef {Object} Session
   * @property {{name:string, initials:string, tint:string}} user
   * @property {Array<{name:string, initials:string, tint:string}>} presence
   *
   * @typedef {Object} Source
   * @property {string} id
   * @property {string} name
   * @property {"live"|"syncing"|"error"} status
   * @property {string} [updatedAt]
   *
   * @typedef {Object} InvestigationSummary
   * @property {string} id
   * @property {string} title
   * @property {number} sourceCount
   * @property {string} updatedAt
   *
   * @typedef {Object} Provenance
   * @property {string} sourceId
   * @property {string} title
   * @property {string} publisher
   * @property {string} coverage
   * @property {string} sourceWatermark  // ISO date the upstream data last changed
   * @property {number} rowCount
   * @property {string} url
   *
   * @typedef {Object} QueryResult
   * @property {"ran"|"stale"|"error"} status
   * @property {"bar"|"table"} view
   * @property {string} title
   * @property {string} unit
   * @property {Array<{label:string, value:number}>} series   // for the chart
   * @property {{columns:Array<{key:string,label:string,type:string}>, rows:Object[]}} table
   * @property {?{label:string, value:number}} delta          // headline callout
   * @property {Array<{dim:string, note:string}>} gaps        // honest-gap flags
   * @property {string} materializedAt
   * @property {Provenance} provenance
   *
   * @typedef {Object} Block
   * @property {string} id
   * @property {"prose"|"query"|"source"} type
   * @property {number} position
   * @property {string} [text]                 // prose
   * @property {string} [nl]                    // query: natural-language prompt
   * @property {?string} [sql]                  // query: AI-compiled SQL
   * @property {?QueryResult} [result]          // query: last run
   * @property {Provenance} [provenance]        // source block
   *
   * @typedef {Object} Investigation
   * @property {string} id
   * @property {string} title
   * @property {string} updatedAt
   * @property {Block[]} blocks
   */

  /* -------------------------------------------------------------------------
   * Helpers
   * ---------------------------------------------------------------------- */
  function delay(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }
  function uid(p) { return (p || "id") + "_" + Math.random().toString(36).slice(2, 9); }
  function nowISO() { return new Date().toISOString(); }

  /**
   * Thin fetch wrapper used by every live-mode method. Centralizes auth,
   * JSON parsing, and error handling so individual methods stay declarative.
   */
  async function http(method, path, body) {
    var res = await fetch(MQ.config.baseUrl + path, {
      method: method,
      headers: Object.assign(
        { "Content-Type": "application/json" },
        MQ.config.token ? { Authorization: "Bearer " + MQ.config.token } : {}
      ),
      body: body ? JSON.stringify(body) : undefined
    });
    if (!res.ok) throw new Error("MapleQuery API " + res.status + " on " + method + " " + path);
    return res.status === 204 ? null : res.json();
  }

  /* =========================================================================
   * MOCK FIXTURES — deleted the day the backend lands.
   * Kept in one object so the seam stays obvious.
   * ======================================================================= */
  var MOCK = {
    session: {
      user: { name: "You", initials: "YO", tint: "primary" },
      presence: [
        { name: "Dana Okafor", initials: "DO", tint: "tint-sky" },
        { name: "Priya Menon", initials: "PM", tint: "tint-peach" }
      ]
    },

    sources: [
      { id: "src_ogp", name: "Open Government Portal", status: "live", updatedAt: "2024-11-19" },
      { id: "src_statcan", name: "Statistics Canada", status: "live", updatedAt: "2024-11-19" },
      { id: "src_pd", name: "Proactive Disclosure", status: "live", updatedAt: "2024-11-12" },
      { id: "src_obd", name: "Open By Default", status: "syncing", updatedAt: "2024-11-01" }
    ],

    investigations: [
      { id: "inv_it", title: "Federal IT contracting: a six-year trend", updatedAt: "just now", saved: true },
      { id: "inv_housing", title: "Housing grants by province", updatedAt: "2d ago", saved: true },
      { id: "inv_imm", title: "Immigration backlog trends", updatedAt: "5d ago", saved: false },
      { id: "inv_health", title: "Health transfer allocations", updatedAt: "1w ago", saved: false },
      { id: "inv_climate", title: "Climate program funding", updatedAt: "3w ago", saved: false }
    ],

    // Each dataset carries both a chart series and a full table so a query can
    // render either view — and a provenance record so every result is cited.
    datasets: {
      it: {
        id: "it", category: "Spending",
        title: "Federal IT Contract Spend",
        desc: "Annual contracted IT-services spending by federal department.",
        unit: "$B, nominal",
        provenance: {
          sourceId: "src_public_accounts_vol3",
          title: "Public Accounts of Canada, Vol. III — Contracts",
          publisher: "Receiver General for Canada",
          coverage: "2018–2024", sourceWatermark: "2024-10-31",
          rowCount: 14228, url: "https://open.canada.ca/"
        },
        meta: { Coverage: "2018–2024", Updated: "2024-10-31", Rows: "6", Completeness: "98%" },
        series: [
          { label: "18–19", value: 3.1 }, { label: "19–20", value: 3.5 },
          { label: "20–21", value: 4.0 }, { label: "21–22", value: 4.5 },
          { label: "22–23", value: 5.0 }, { label: "23–24", value: 5.4 }
        ],
        columns: [
          { key: "dept", label: "Department", type: "text" },
          { key: "y18", label: "2018–19", type: "money" },
          { key: "y24", label: "2023–24", type: "money" },
          { key: "chg", label: "Change", type: "pct" }
        ],
        rows: [
          { dept: "National Defence", y18: 0.61, y24: 1.18, chg: 93 },
          { dept: "Shared Services Canada", y18: 0.50, y24: 0.90, chg: 81 },
          { dept: "Employment & Social Dev.", y18: 0.33, y24: 0.57, chg: 71 },
          { dept: "Canada Revenue Agency", y18: 0.44, y24: 0.66, chg: 52 },
          { dept: "Public Services & Proc.", y18: 0.30, y24: 0.45, chg: 50 },
          { dept: "Health Canada", y18: 0.19, y24: 0.26, chg: 37 }
        ],
        delta: { value: 74, suffix: "%", label: "nominal increase over six years" },
        gaps: [{ dim: "20–21", note: "2 departments suppressed (<5 contracts, privacy)" }],
        sql: "SELECT fiscal_year,\n       ROUND(SUM(amount_cad) / 1e9, 1) AS spend_b\nFROM   curated.contracts\nWHERE  object_code = 'IT_SERVICES'\n  AND  fiscal_year BETWEEN '2018-19' AND '2023-24'\nGROUP  BY fiscal_year\nORDER  BY fiscal_year;"
      },
      housing: {
        id: "housing", category: "Housing",
        title: "Housing Grant Approvals by Province",
        desc: "CMHC housing-program funding approvals by province.",
        unit: "$M approved",
        provenance: {
          sourceId: "src_cmhc", title: "CMHC Housing Program Approvals",
          publisher: "Canada Mortgage and Housing Corp.", coverage: "2020–2024",
          sourceWatermark: "2024-08-15", rowCount: 5120, url: "https://open.canada.ca/"
        },
        meta: { Coverage: "2020–2024", Updated: "2024-08-15", Rows: "6", Completeness: "95%" },
        series: [
          { label: "ON", value: 5980 }, { label: "QC", value: 3360 },
          { label: "BC", value: 3120 }, { label: "AB", value: 1840 },
          { label: "NS", value: 540 }, { label: "MB", value: 430 }
        ],
        columns: [
          { key: "prov", label: "Province", type: "text" },
          { key: "approved", label: "Approved ($M)", type: "num" },
          { key: "units", label: "Units", type: "int" },
          { key: "per10k", label: "Per 10K homes", type: "int" }
        ],
        rows: [
          { prov: "British Columbia", approved: 3120, units: 18400, per10k: 41 },
          { prov: "Ontario", approved: 5980, units: 42100, per10k: 37 },
          { prov: "Nova Scotia", approved: 540, units: 3250, per10k: 34 },
          { prov: "Quebec", approved: 3360, units: 28700, per10k: 31 },
          { prov: "Alberta", approved: 1840, units: 14200, per10k: 28 },
          { prov: "Manitoba", approved: 430, units: 3900, per10k: 25 }
        ],
        delta: { value: 14.2, dec: 1, prefix: "$", suffix: "B", label: "approved nationwide, 2020–2024" },
        gaps: [{ dim: "Territories", note: "YT / NT / NU excluded — reported separately" }],
        sql: "SELECT province,\n       ROUND(SUM(amount_cad) / 1e6) AS approved_m,\n       SUM(units)              AS units\nFROM   curated.cmhc_approvals\nWHERE  program_year BETWEEN 2020 AND 2024\nGROUP  BY province\nORDER  BY approved_m DESC;"
      },
      imm: {
        id: "imm", category: "Immigration",
        title: "Immigration PR Processing Inventory",
        desc: "Permanent-residence applications in the processing inventory by year.",
        unit: "M applications",
        provenance: {
          sourceId: "src_ircc", title: "IRCC Processing Times — Permanent Residence",
          publisher: "Immigration, Refugees and Citizenship Canada", coverage: "2019–2024",
          sourceWatermark: "2024-11-01", rowCount: 72, url: "https://open.canada.ca/"
        },
        meta: { Coverage: "2019–2024", Updated: "2024-11-01", Rows: "6", Completeness: "88%" },
        series: [
          { label: "2019", value: 1.5 }, { label: "2020", value: 1.6 },
          { label: "2021", value: 1.8 }, { label: "2022", value: 2.2 },
          { label: "2023", value: 1.6 }, { label: "2024", value: 1.1 }
        ],
        columns: [
          { key: "year", label: "Year", type: "text" },
          { key: "inv", label: "Inventory (M)", type: "big" },
          { key: "yoy", label: "YoY change", type: "pct" },
          { key: "status", label: "Status", type: "text" }
        ],
        rows: [
          { year: "2019", inv: 1.5, yoy: 0, status: "baseline" },
          { year: "2020", inv: 1.6, yoy: 7, status: "rising" },
          { year: "2021", inv: 1.8, yoy: 13, status: "rising" },
          { year: "2022", inv: 2.2, yoy: 22, status: "peak" },
          { year: "2023", inv: 1.6, yoy: -27, status: "falling" },
          { year: "2024", inv: 1.1, yoy: -31, status: "recovering" }
        ],
        delta: { value: 50, suffix: "%", label: "down from the 2022 peak" },
        gaps: [{ dim: "2024", note: "Partial year — Q4 estimated from October data" }],
        sql: "SELECT year,\n       ROUND(inventory / 1e6, 1) AS inventory_m,\n       yoy_change_pct\nFROM   curated.ircc_pr_inventory\nWHERE  year BETWEEN 2019 AND 2024\nORDER  BY year;"
      }
    },

    // Ask-tab knowledge base (RAG stand-in).
    answers: [
      { k: ["it", "contract", "spend", "technology"], dataset: "it",
        a: "Federal spending on contracted IT services grew from <strong>$3.1B</strong> in 2018–19 to <strong>$5.4B</strong> in 2023–24 — about a <strong>74% increase</strong>. National Defence grew the most." },
      { k: ["housing", "grant", "cmhc", "province"], dataset: "housing",
        a: "Between 2020 and 2024, CMHC approved roughly <strong>$14.2B</strong> in housing-program funding. Ontario and Quebec approved the most overall; B.C. leads per-capita." },
      { k: ["immigration", "backlog", "processing", "ircc", "residence", "pr"], dataset: "imm",
        a: "The permanent-residence inventory peaked near <strong>2.2M</strong> in 2022 and fell to roughly <strong>1.1M</strong> by late 2024 — about <strong>50% down</strong> from the peak." }
    ]
  };

  /** Pick the dataset a natural-language prompt is "about" (mock NL routing). */
  function routeQuery(nl) {
    var q = (nl || "").toLowerCase();
    var best = "it", score = 0;
    Object.keys(MOCK.datasets).forEach(function (id) {
      var hit = MOCK.answers.find(function (a) { return a.dataset === id; });
      var s = hit ? hit.k.reduce(function (n, w) { return n + (q.indexOf(w) > -1 ? 1 : 0); }, 0) : 0;
      if (s > score) { score = s; best = id; }
    });
    return MOCK.datasets[best];
  }

  /** Shape a dataset into the QueryResult contract. */
  function toResult(ds, opts) {
    opts = opts || {};
    return {
      status: opts.stale ? "stale" : "ran",
      view: "bar",
      title: ds.title + " (" + ds.unit + ")",
      unit: ds.unit,
      series: ds.series,
      table: { columns: ds.columns, rows: ds.rows },
      delta: ds.delta,
      gaps: ds.gaps || [],
      materializedAt: nowISO(),
      provenance: ds.provenance
    };
  }

  /* =========================================================================
   * PUBLIC API — identical surface for mock + live.
   * ======================================================================= */
  MQ.api = {
    /** @returns {Promise<Session>}  @endpoint GET /session */
    async getSession() {
      if (MQ.config.mode === "live") return http("GET", "/session");
      await delay(MQ.config.latency.fast);
      return MOCK.session;
    },

    /** @returns {Promise<InvestigationSummary[]>}  @endpoint GET /investigations */
    async listInvestigations() {
      if (MQ.config.mode === "live") return http("GET", "/investigations");
      await delay(MQ.config.latency.fast);
      return MOCK.investigations.slice();
    },

    /** @returns {Promise<Investigation>}  @endpoint GET /investigations/:id */
    async getInvestigation(id) {
      if (MQ.config.mode === "live") return http("GET", "/investigations/" + id);
      await delay(MQ.config.latency.base);
      id = id || "inv_it";
      var seeds = {
        inv_it: {
          title: "Federal IT contracting: a six-year trend",
          blocks: [
            { type: "prose", text: "Federal departments have leaned harder on outside IT contractors since the pandemic. To size the shift, I pulled contract spend from the Public Accounts and asked MapleQuery to track it over six fiscal years." },
            { type: "query", nl: "Total federal IT-services contract spend by fiscal year, 2018–2024", sql: null, result: null },
            { type: "prose", text: "The next question is whether the growth is concentrated — so I broke it down by department." }
          ]
        },
        inv_housing: {
          title: "Housing grants by province",
          blocks: [
            { type: "prose", text: "CMHC housing-program approvals vary widely by province. I wanted the raw distribution before drawing any per-capita conclusions." },
            { type: "query", nl: "CMHC housing grant approvals by province, 2020–2024", sql: null, result: null }
          ]
        },
        inv_imm: {
          title: "Immigration backlog trends",
          blocks: [
            { type: "prose", text: "The permanent-residence inventory ballooned during the pandemic. Here is how the processing backlog moved year over year." },
            { type: "query", nl: "Immigration PR processing inventory by year, 2019–2024", sql: null, result: null }
          ]
        }
      };
      var seed = seeds[id] || { title: "Untitled investigation", blocks: [{ type: "prose", text: "" }] };
      return {
        id: id,
        title: seed.title,
        updatedAt: "just now",
        blocks: seed.blocks.map(function (b, i) { return Object.assign({ id: uid("blk"), position: i }, b); })
      };
    },

    /** @returns {Promise<Source[]>}  @endpoint GET /sources */
    async listSources() {
      if (MQ.config.mode === "live") return http("GET", "/sources");
      await delay(MQ.config.latency.fast);
      return MOCK.sources.slice();
    },

    /**
     * AI step: translate a natural-language prompt into SQL WITHOUT running it.
     * Mirrors an agent "plan" phase so the UI can show the compiled query.
     * @returns {Promise<{sql:string, datasetRef:string}>}
     * @endpoint POST /investigations/:id/blocks/:blockId:compile
     */
    async compileQuery(payload) {
      if (MQ.config.mode === "live")
        return http("POST", "/investigations/" + payload.investigationId +
          "/blocks/" + payload.blockId + ":compile", { nl: payload.nl });
      await delay(MQ.config.latency.query * 0.5);
      var ds = routeQuery(payload.nl);
      return { sql: ds.sql, datasetRef: "curated." + ds.id };
    },

    /**
     * Execute a query block and return a fully-cited result.
     * @returns {Promise<QueryResult>}
     * @endpoint POST /investigations/:id/blocks/:blockId:run
     */
    async runQuery(payload) {
      if (MQ.config.mode === "live")
        return http("POST", "/investigations/" + payload.investigationId +
          "/blocks/" + payload.blockId + ":run", { nl: payload.nl });
      await delay(MQ.config.latency.query);
      return toResult(routeQuery(payload.nl));
    },

    /**
     * Ask-tab agent. In live mode you'd likely stream tokens; here we resolve
     * once with the answer + citation.
     * @returns {Promise<{answer:?string, citation:?Provenance}>}
     * @endpoint POST /agent/ask
     */
    async askAgent(question) {
      if (MQ.config.mode === "live") return http("POST", "/agent/ask", { question: question });
      await delay(MQ.config.latency.agent);
      var q = (question || "").toLowerCase();
      var hit = MOCK.answers.find(function (a) {
        return a.k.some(function (w) { return q.indexOf(w) > -1; });
      });
      if (!hit) return { answer: null, citation: null };
      return { answer: hit.a, citation: MOCK.datasets[hit.dataset].provenance };
    },

    /** @returns {Promise<Array>}  @endpoint GET /datasets */
    async listDatasets() {
      if (MQ.config.mode === "live") return http("GET", "/datasets");
      await delay(MQ.config.latency.fast);
      return Object.keys(MOCK.datasets).map(function (id) {
        var d = MOCK.datasets[id];
        return { id: id, category: d.category, title: d.title, rows: d.rows.length };
      });
    },

    /** @returns {Promise<Object>}  @endpoint GET /datasets/:id */
    async getDataset(id) {
      if (MQ.config.mode === "live") return http("GET", "/datasets/" + id);
      await delay(MQ.config.latency.fast);
      return MOCK.datasets[id] || MOCK.datasets.it;
    },

    /** @returns {Promise<{sourceCount:number, sections:number}>}  @endpoint POST /investigations/:id:promote */
    async promoteToReport(payload) {
      if (MQ.config.mode === "live")
        return http("POST", "/investigations/" + payload.investigationId + ":promote", payload);
      await delay(MQ.config.latency.base);
      return { sourceCount: payload.sourceCount || 1, sections: payload.blockCount || 3 };
    },

    /* ---- Block persistence (fire-and-forget in the mock) ------------------ */
    /** @endpoint POST /investigations/:id/blocks */
    async createBlock(payload) {
      if (MQ.config.mode === "live")
        return http("POST", "/investigations/" + payload.investigationId + "/blocks", payload.block);
      await delay(80); return Object.assign({ id: uid("blk") }, payload.block);
    },
    /** @endpoint PATCH /investigations/:id/blocks/:blockId */
    async updateBlock(payload) {
      if (MQ.config.mode === "live")
        return http("PATCH", "/investigations/" + payload.investigationId + "/blocks/" + payload.blockId, payload.patch);
      await delay(40); return { ok: true };
    },
    /** @endpoint PATCH /investigations/:id/blocks:reorder */
    async reorderBlocks(payload) {
      if (MQ.config.mode === "live")
        return http("PATCH", "/investigations/" + payload.investigationId + "/blocks:reorder", { order: payload.order });
      await delay(40); return { ok: true };
    },
    /** @endpoint DELETE /investigations/:id/blocks/:blockId */
    async deleteBlock(payload) {
      if (MQ.config.mode === "live")
        return http("DELETE", "/investigations/" + payload.investigationId + "/blocks/" + payload.blockId);
      await delay(40); return { ok: true };
    }
  };

})(window);
