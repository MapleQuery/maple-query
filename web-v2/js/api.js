/* ============================================================================
   MapleQuery ATLAS — api.js
   ----------------------------------------------------------------------------
   THE BACKEND SEAM. Production-ready SSE streaming client + REST surface.

   Handles ALL 18 event types from agent-service:
     turn_start, cache_hit, retrieval_started, datasets_ranked,
     columns_ranked, sample_rows, sql_generated, sql_guarded,
     sql_executed, rows, message_delta, cost_update, budget_exceeded,
     turn_timeout, tool_error, turn_record, derivation, done, error

   Also exposes REST endpoints:
     listDatasets, getDataset, getDatasetColumns, getDatasetDocuments,
     runSql, getCorpusStats

   Config comes from MQ.config (config.js). When MQ.config.mode === "mock",
   falls back to local fixtures for offline design work.
   ========================================================================== */
(function (global) {
  "use strict";
  var MQ = global.MQ || (global.MQ = {});

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------
  function uid(p) { return (p || "id") + "_" + Math.random().toString(36).slice(2, 9); }
  function nowISO() { return new Date().toISOString(); }

  /**
   * Thin fetch wrapper for REST endpoints.
   */
  async function jsonFetch(method, path, body, signal) {
    var cfg = MQ.config;
    var headers = Object.assign(
      { "Content-Type": "application/json" },
      cfg.authHeaders()
    );
    var opts = { method: method, headers: headers, signal: signal || undefined };
    if (body) opts.body = JSON.stringify(body);

    var res = await fetch(cfg.baseUrl + path, opts);
    var text = await res.text();
    if (!res.ok) {
      var err = new Error("API " + res.status + " on " + method + " " + path);
      err.status = res.status;
      err.body = text;
      throw err;
    }
    return text ? JSON.parse(text) : null;
  }

  // ---------------------------------------------------------------------------
  // SSE Streaming — mirrors production's lib/sse.ts
  // ---------------------------------------------------------------------------

  /**
   * Known event names. If the server sends an unknown event we skip it.
   */
  // Mirrors EventType in semantic_enrich/core/agent_events.py (27 types).
  var KNOWN_EVENTS = [
    "turn_start", "cache_hit", "retrieval_started", "datasets_ranked",
    "columns_ranked", "documents_listed", "sample_rows", "sql_generated",
    "sql_guarded", "sql_executed", "rows", "message_delta", "cost_update",
    "budget_exceeded", "turn_timeout", "tool_error", "done", "error",
    "phase_start", "triage_result", "reformulation", "verification",
    "plan_hint", "turn_record", "derivation", "suggestions"
  ];
  var KNOWN_SET = {};
  KNOWN_EVENTS.forEach(function (n) { KNOWN_SET[n] = true; });

  /**
   * Stream POST /chat with typed event dispatch.
   *
   * Uses native EventSource-style parsing over fetch() ReadableStream
   * (no external dependency like @microsoft/fetch-event-source).
   *
   * @param {Object} request - { conversation_id, question, history, turn_records }
   * @param {Object} handlers - { onEvent, onDone, onError, onOpen, onMalformed }
   * @param {AbortSignal} [signal] - optional abort signal
   * @returns {Promise<void>}
   */
  async function streamChat(request, handlers, signal) {
    var cfg = MQ.config;
    var sawDone = false;

    var res;
    try {
      res = await fetch(cfg.baseUrl + "/chat", {
        method: "POST",
        headers: Object.assign(
          { "Content-Type": "application/json", "Accept": "text/event-stream" },
          cfg.authHeaders()
        ),
        body: JSON.stringify(request),
        signal: signal || undefined
      });
    } catch (err) {
      if (err.name === "AbortError") return;
      if (handlers.onError) handlers.onError({ message: String(err), retryable: true });
      return;
    }

    if (!res.ok) {
      var errText = "";
      try { errText = await res.text(); } catch (e) {}
      if (handlers.onError) {
        handlers.onError({
          message: "chat " + res.status + ": " + errText,
          retryable: res.status >= 500
        });
      }
      return;
    }

    if (handlers.onOpen) handlers.onOpen();

    // Parse SSE from the response body stream
    var reader = res.body.getReader();
    var decoder = new TextDecoder();
    var buffer = "";
    var currentEvent = "";
    var currentData = "";

    try {
      while (true) {
        var result = await reader.read();
        if (result.done) break;

        buffer += decoder.decode(result.value, { stream: true });

        // Process complete lines
        var lines = buffer.split("\n");
        buffer = lines.pop(); // keep incomplete line in buffer

        for (var i = 0; i < lines.length; i++) {
          var line = lines[i];

          if (line === "") {
            // Empty line = end of event
            if (currentData) {
              dispatchEvent(currentEvent || "message", currentData, handlers);
              if ((currentEvent || "message") === "done") sawDone = true;
            }
            currentEvent = "";
            currentData = "";
          } else if (line.charAt(0) === ":") {
            // Comment line (keepalive), ignore
          } else if (line.indexOf("event:") === 0) {
            currentEvent = line.slice(6).trim();
          } else if (line.indexOf("data:") === 0) {
            var dataLine = line.slice(5);
            // SSE spec: if first char after "data:" is space, skip it
            if (dataLine.charAt(0) === " ") dataLine = dataLine.slice(1);
            currentData += (currentData ? "\n" : "") + dataLine;
          }
        }
      }

      // Handle any remaining buffered event
      if (currentData) {
        dispatchEvent(currentEvent || "message", currentData, handlers);
        if ((currentEvent || "message") === "done") sawDone = true;
      }
    } catch (err) {
      if (err.name === "AbortError") return;
      if (handlers.onError) {
        handlers.onError({ message: String(err), retryable: true });
      }
      return;
    }

    // Stream closed without done event
    if (!sawDone && handlers.onError) {
      handlers.onError({ message: "stream closed before done", retryable: true });
    }
  }

  function dispatchEvent(name, rawData, handlers) {
    if (!KNOWN_SET[name]) return;

    var payload;
    try {
      payload = JSON.parse(rawData);
    } catch (err) {
      if (handlers.onMalformed) handlers.onMalformed(name, rawData, err);
      return;
    }

    // Dispatch typed event
    if (handlers.onEvent) {
      handlers.onEvent({ name: name, payload: payload });
    }

    // Terminal events
    if (name === "done" && handlers.onDone) {
      handlers.onDone(payload);
    }
    if (name === "error" && handlers.onError) {
      handlers.onError(payload);
    }
  }

  // ---------------------------------------------------------------------------
  // REST API — mirrors production's lib/api.ts
  // ---------------------------------------------------------------------------

  /**
   * @param {Object} params - { q?, limit?, offset?, signal? }
   * @returns {Promise<{ datasets: DatasetSummary[], total: number }>}
   */
  function listDatasets(params) {
    params = params || {};
    var qs = new URLSearchParams();
    if (params.q) qs.set("q", params.q);
    if (params.limit != null) qs.set("limit", String(params.limit));
    if (params.offset != null) qs.set("offset", String(params.offset));
    var suffix = qs.toString() ? "?" + qs.toString() : "";
    return jsonFetch("GET", "/datasets" + suffix, null, params.signal);
  }

  function getDataset(packageId, signal) {
    return jsonFetch("GET", "/datasets/" + encodeURIComponent(packageId), null, signal);
  }

  function getDatasetColumns(packageId, signal) {
    return jsonFetch("GET", "/datasets/" + encodeURIComponent(packageId) + "/columns", null, signal);
  }

  function getDatasetDocuments(packageId, signal) {
    return jsonFetch("GET", "/datasets/" + encodeURIComponent(packageId) + "/documents", null, signal);
  }

  /**
   * Direct SQL execution (goes through guard).
   * @param {string} sql
   * @param {string} [rationale]
   * @param {AbortSignal} [signal]
   * @returns {Promise<SqlRunResponse>}
   */
  function runSql(sql, rationale, signal) {
    return jsonFetch("POST", "/sql/run", { sql: sql, rationale: rationale || "" }, signal);
  }

  /**
   * Corpus stats for the landing page hero.
   */
  function getCorpusStats(signal) {
    return jsonFetch("GET", "/corpus/stats", null, signal);
  }

  // ---------------------------------------------------------------------------
  // Mock fixtures (for offline design work when mode === "mock")
  // ---------------------------------------------------------------------------

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
    corpusStats: { datasets: 47, documents: 312, rows: 14200000 }
  };

  function delay(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }

  /**
   * Mock streaming — simulates the full event sequence for offline work.
   */
  async function mockStreamChat(request, handlers) {
    if (handlers.onOpen) handlers.onOpen();
    await delay(200);

    var turnId = uid("turn");
    var conversationId = request.conversation_id;

    // turn_start
    handlers.onEvent({ name: "turn_start", payload: {
      conversation_id: conversationId, turn_id: turnId, cached: false
    }});
    await delay(300);

    // retrieval_started
    handlers.onEvent({ name: "retrieval_started", payload: {
      query: request.question, k: 5
    }});
    await delay(400);

    // datasets_ranked
    handlers.onEvent({ name: "datasets_ranked", payload: {
      candidates: [
        { package_id: "public_accounts_contracts", title: "Public Accounts — Contracts", summary: "Federal contract spending by department", distance: 0.12 },
        { package_id: "cmhc_approvals", title: "CMHC Housing Approvals", summary: "Housing program funding by province", distance: 0.34 }
      ]
    }});
    await delay(300);

    // columns_ranked
    handlers.onEvent({ name: "columns_ranked", payload: {
      package_ids: ["public_accounts_contracts"],
      candidates: [
        { package_id: "public_accounts_contracts", column_name: "amount_cad", description: "Contract value in CAD", distance: 0.08 },
        { package_id: "public_accounts_contracts", column_name: "fiscal_year", description: "Fiscal year (YYYY-YY)", distance: 0.15 },
        { package_id: "public_accounts_contracts", column_name: "department", description: "Federal department name", distance: 0.22 }
      ]
    }});
    await delay(200);

    // sql_generated
    var sql = "SELECT fiscal_year,\n       ROUND(SUM(amount_cad) / 1e9, 1) AS spend_b\nFROM   curated.contracts\nWHERE  object_code = 'IT_SERVICES'\n  AND  fiscal_year BETWEEN '2018-19' AND '2023-24'\nGROUP  BY fiscal_year\nORDER  BY fiscal_year;";
    handlers.onEvent({ name: "sql_generated", payload: {
      sql: sql,
      rationale: "Sum IT-services contracts by fiscal year from Public Accounts"
    }});
    await delay(400);

    // sql_guarded
    handlers.onEvent({ name: "sql_guarded", payload: {
      accepted: true,
      reason: null,
      sql_final: sql,
      dry_run_bytes: 4200000
    }});
    await delay(300);

    // sql_executed
    var rows = [
      { fiscal_year: "2018-19", spend_b: 3.1 },
      { fiscal_year: "2019-20", spend_b: 3.5 },
      { fiscal_year: "2020-21", spend_b: 4.0 },
      { fiscal_year: "2021-22", spend_b: 4.5 },
      { fiscal_year: "2022-23", spend_b: 5.0 },
      { fiscal_year: "2023-24", spend_b: 5.4 }
    ];
    handlers.onEvent({ name: "sql_executed", payload: {
      row_count: 6,
      bytes_billed: 4200000,
      elapsed_ms: 820,
      sample_rows: rows
    }});
    await delay(200);

    // derivation
    handlers.onEvent({ name: "derivation", payload: {
      dataset_titles: ["Public Accounts — Contracts"],
      source_packages: ["public_accounts_contracts"],
      aggregation: "SUM",
      value_columns: ["amount_cad"],
      scope: "IT_SERVICES, 2018-2024",
      row_count: 6,
      source_row_estimate: 14228,
      result_value: 5400000000,
      result_label: "Total IT spend 2023-24",
      unit_scale: "dollars",
      unit_source: "amount_cad",
      flags: []
    }});
    await delay(100);

    // cost_update
    handlers.onEvent({ name: "cost_update", payload: {
      tokens_in_total: 2840,
      tokens_out_total: 680,
      dollars_spent: 0.012
    }});

    // message_delta (stream the answer text in chunks)
    var answer = "Federal spending on contracted IT services grew from **$3.1B** in 2018\u201319 to **$5.4B** in 2023\u201324 \u2014 about a **74% increase**. National Defence grew the most, followed by Shared Services Canada.";
    var chunks = answer.match(/.{1,20}/g) || [answer];
    for (var i = 0; i < chunks.length; i++) {
      handlers.onEvent({ name: "message_delta", payload: { delta: chunks[i] }});
      await delay(30);
    }
    await delay(100);

    // turn_record
    handlers.onEvent({ name: "turn_record", payload: {
      record: {
        turn_id: turnId,
        question: request.question,
        datasets_used: ["public_accounts_contracts"],
        sql_ran: sql
      }
    }});

    // done
    var donePayload = {
      turn_id: turnId,
      total_tool_calls: 3,
      total_dollars: 0.012,
      elapsed_ms: 2840
    };
    handlers.onEvent({ name: "done", payload: donePayload });
    if (handlers.onDone) handlers.onDone(donePayload);
  }

  // ---------------------------------------------------------------------------
  // Public API surface
  // ---------------------------------------------------------------------------

  MQ.api = {
    // Unique ID generator (exposed for store/UI)
    uid: uid,

    /**
     * Stream a chat question. Dispatches typed events via handlers.
     *
     * @param {Object} request - { conversation_id, question, history, turn_records }
     * @param {Object} handlers - { onEvent, onDone, onError, onOpen, onMalformed }
     * @param {AbortSignal} [signal]
     * @returns {Promise<void>}
     */
    streamChat: function (request, handlers, signal) {
      if (MQ.config.mode === "mock") {
        return mockStreamChat(request, handlers);
      }
      return streamChat(request, handlers, signal);
    },

    // REST endpoints
    listDatasets: function (params) {
      if (MQ.config.mode === "mock") {
        return delay(300).then(function () {
          return { datasets: [], total: 0 };
        });
      }
      return listDatasets(params);
    },

    getDataset: function (packageId, signal) {
      if (MQ.config.mode === "mock") {
        return delay(200).then(function () { return { package_id: packageId, title: packageId, summary: "Mock dataset" }; });
      }
      return getDataset(packageId, signal);
    },

    getDatasetColumns: function (packageId, signal) {
      if (MQ.config.mode === "mock") {
        return delay(200).then(function () { return { package_id: packageId, columns: [] }; });
      }
      return getDatasetColumns(packageId, signal);
    },

    getDatasetDocuments: function (packageId, signal) {
      if (MQ.config.mode === "mock") {
        return delay(200).then(function () { return { package_id: packageId, documents: [] }; });
      }
      return getDatasetDocuments(packageId, signal);
    },

    runSql: function (sql, rationale, signal) {
      if (MQ.config.mode === "mock") {
        return delay(600).then(function () {
          return { status: "ok", reason: null, sql_final: sql, rows: [], row_count: 0, bytes_billed: null, elapsed_ms: 420, truncated: false };
        });
      }
      return runSql(sql, rationale, signal);
    },

    getCorpusStats: function (signal) {
      if (MQ.config.mode === "mock") {
        return delay(200).then(function () { return MOCK.corpusStats; });
      }
      return getCorpusStats(signal);
    },

    // Session/sources (mock-only for now — auth flow will replace)
    getSession: function () {
      return delay(200).then(function () { return MOCK.session; });
    },
    listSources: function () {
      if (MQ.config.mode === "mock") {
        return delay(200).then(function () { return MOCK.sources; });
      }
      return jsonFetch("GET", "/sources");
    }
  };

})(window);
