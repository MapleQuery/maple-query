/* ============================================================================
   MapleQuery ATLAS — store.js
   ----------------------------------------------------------------------------
   Reactive state container. Handles ALL event types from the agent-service
   SSE stream and maintains conversation/notebook/explorer state.

   Surfaces:
     - Chat stream state (evidence cards, assistant text, cost, status)
     - Conversation history (multi-turn with turn_records)
     - Notebook blocks (prose + query)
     - Explorer steps (prompt → SQL → rows)
     - Document/investigation state (Stage blocks)
     - Theme, session, sources

   app.js (atlas.js) subscribes to state changes and renders the DOM.
   ========================================================================== */
(function (global) {
  "use strict";
  var MQ = global.MQ || (global.MQ = {});

  function uid(p) { return (p || "id") + "_" + Math.random().toString(36).slice(2, 9); }
  function nowISO() { return new Date().toISOString(); }

  function createStore() {
    var state = {
      // --- Theme / session ---
      theme: "light",
      session: null,
      sources: [],

      // --- Stage (investigation document) ---
      investigation: null,
      blocks: [],

      // --- Chat streaming state (mirrors use-chat-stream.ts reducer) ---
      stream: {
        status: "idle",        // idle | streaming | done | error
        cards: [],             // evidence rail cards
        assistantText: "",     // accumulated message_delta
        toolCalls: 0,
        dollars: 0,
        elapsedMs: null,
        cached: false,
        error: null,
        turnRecord: null,
        suggestions: []        // next-step offers from the backend
      },

      // --- Conversation history ---
      conversation: null,      // current StoredConversation
      conversationIndex: [],   // list of { id, title, updatedAt }

      // --- Notebook ---
      notebook: null,          // current StoredNotebook
      notebookIndex: [],

      // --- Explorer ---
      explorer: null,          // current StoredExplorer
      explorerIndex: [],

      // --- Datasets ---
      datasets: [],
      datasetsTotal: 0,
      datasetsLoading: false,
      datasetDetail: null,
      datasetColumns: null,
      datasetDocuments: null
    };

    var subs = { "*": [] };

    function emit(evt) {
      (subs[evt] || []).forEach(function (fn) { fn(state); });
      subs["*"].forEach(function (fn) { fn(state, evt); });
    }

    // =========================================================================
    // Stream reducer — processes each SSE event into state.stream
    // Mirrors production's use-chat-stream.ts reducer exactly
    // =========================================================================

    function streamReset() {
      state.stream = {
        status: "idle", cards: [], assistantText: "",
        toolCalls: 0, dollars: 0, elapsedMs: null,
        cached: false, error: null, turnRecord: null, suggestions: []
      };
      emit("stream");
    }

    function streamStart() {
      state.stream = {
        status: "streaming", cards: [], assistantText: "",
        toolCalls: 0, dollars: 0, elapsedMs: null,
        cached: false, error: null, turnRecord: null, suggestions: []
      };
      emit("stream");
    }

    function streamError(message) {
      state.stream.status = "error";
      state.stream.error = message;
      emit("stream");
    }

    function upsertBySqlId(cards, mutator) {
      for (var i = cards.length - 1; i >= 0; i--) {
        var next = mutator(cards[i]);
        if (next) { cards[i] = next; return cards; }
      }
      return cards;
    }

    function streamEvent(event) {
      var s = state.stream;
      var name = event.name;
      var p = event.payload;

      switch (name) {
        case "turn_start":
          s.cached = p.cached;
          s.status = "streaming";
          break;

        case "phase_start":
          // Informational — no state change needed
          break;

        case "triage_result":
          // Informational — no state change needed
          break;

        case "cache_hit":
          s.cached = true;
          break;

        case "retrieval_started":
          s.cards.push({ id: uid("c"), kind: "retrieval_started", query: p.query, k: p.k });
          break;

        case "datasets_ranked":
          s.cards.push({ id: uid("c"), kind: "datasets_ranked", candidates: p.candidates });
          break;

        case "columns_ranked":
          s.cards.push({ id: uid("c"), kind: "columns_ranked", packageIds: p.package_ids, candidates: p.candidates });
          break;

        case "sample_rows":
          s.cards.push({ id: uid("c"), kind: "sample_rows", packageId: p.package_id, rows: p.rows });
          break;

        case "derivation":
          s.cards.push({ id: uid("c"), kind: "derivation", derivation: p });
          break;

        case "sql_generated":
          s.cards.push({ id: uid("c"), kind: "sql_generated", sql: p.sql, rationale: p.rationale });
          break;

        case "sql_guarded":
          upsertBySqlId(s.cards, function (c) {
            if (c.kind === "sql_generated" && !c.guard) {
              c.guard = { accepted: p.accepted, reason: p.reason, sql_final: p.sql_final };
              return c;
            }
            return null;
          });
          break;

        case "sql_executed":
          upsertBySqlId(s.cards, function (c) {
            if (c.kind === "sql_generated" && !c.executed) {
              c.executed = { row_count: p.row_count, elapsed_ms: p.elapsed_ms || null, rows: p.sample_rows || [] };
              return c;
            }
            return null;
          });
          break;

        case "rows":
          upsertBySqlId(s.cards, function (c) {
            if (c.kind === "sql_generated" && c.executed) {
              c.executed.rows = c.executed.rows.concat(p.rows);
              return c;
            }
            return null;
          });
          break;

        case "message_delta":
          s.assistantText += p.delta;
          break;

        case "cost_update":
          s.dollars = p.dollars_spent;
          break;

        case "tool_error":
          s.cards.push({ id: uid("c"), kind: "tool_error", tool: p.tool, message: p.message });
          break;

        case "budget_exceeded":
          s.cards.push({ id: uid("c"), kind: "budget_exceeded", which: p.which, value: p.value, cap: p.cap });
          break;

        case "turn_timeout":
          s.cards.push({ id: uid("c"), kind: "turn_timeout", elapsed_ms: p.elapsed_ms, cap_ms: p.cap_ms });
          break;

        case "turn_record":
          s.turnRecord = p.record;
          break;

        case "suggestions":
          s.suggestions = p.items || [];
          break;

        case "done":
          s.status = "done";
          s.toolCalls = p.total_tool_calls;
          s.dollars = p.total_dollars;
          s.elapsedMs = p.elapsed_ms;
          break;

        case "error":
          s.status = "error";
          s.error = p.message;
          break;
      }

      emit("stream");
    }

    // =========================================================================
    // Conversation management
    // =========================================================================

    function loadConversationIndex() {
      state.conversationIndex = MQ.storage.conversations.list();
      emit("conversationIndex");
    }

    function loadConversation(id) {
      var conv = MQ.storage.conversations.load(id);
      if (!conv) {
        conv = {
          id: id, title: "New conversation",
          createdAt: nowISO(), updatedAt: nowISO(),
          history: [], evidenceByTurnId: {}, turnRecords: []
        };
      }
      state.conversation = conv;
      emit("conversation");
      return conv;
    }

    function saveConversation(conv) {
      conv.updatedAt = nowISO();
      state.conversation = conv;
      MQ.storage.conversations.save(conv);
      loadConversationIndex();
      emit("conversation");
    }

    function appendToHistory(role, content, extras) {
      if (!state.conversation) return;
      var msg = { role: role, content: content };
      if (extras) Object.assign(msg, extras);
      state.conversation.history.push(msg);
    }

    // =========================================================================
    // Notebook management
    // =========================================================================

    function loadNotebookIndex() {
      state.notebookIndex = MQ.storage.notebooks.list();
      emit("notebookIndex");
    }

    function loadNotebook(id) {
      var nb = MQ.storage.notebooks.load(id);
      if (!nb) {
        nb = { id: id, title: "Untitled notebook", createdAt: nowISO(), updatedAt: nowISO(), blocks: [] };
      }
      state.notebook = nb;
      emit("notebook");
      return nb;
    }

    function saveNotebook(nb) {
      nb.updatedAt = nowISO();
      state.notebook = nb;
      MQ.storage.notebooks.save(nb);
      loadNotebookIndex();
      emit("notebook");
    }

    // =========================================================================
    // Explorer management
    // =========================================================================

    function loadExplorerIndex() {
      state.explorerIndex = MQ.storage.explorer.list();
      emit("explorerIndex");
    }

    function loadExplorer(id) {
      var exp = MQ.storage.explorer.load(id);
      if (!exp) {
        exp = { id: id, title: "Explorer session", createdAt: nowISO(), updatedAt: nowISO(), steps: [], activeStepId: null };
      }
      state.explorer = exp;
      emit("explorer");
      return exp;
    }

    function saveExplorer(exp) {
      exp.updatedAt = nowISO();
      state.explorer = exp;
      MQ.storage.explorer.save(exp);
      loadExplorerIndex();
      emit("explorer");
    }

    // =========================================================================
    // Stage blocks (investigation document)
    // =========================================================================

    function reindex(list) { list.forEach(function (b, i) { b.position = i; }); }

    // =========================================================================
    // Public store interface
    // =========================================================================

    return {
      get: function () { return state; },
      uid: uid,

      on: function (evt, fn) {
        (subs[evt] || (subs[evt] = [])).push(fn);
        return function off() { subs[evt] = (subs[evt] || []).filter(function (f) { return f !== fn; }); };
      },

      emit: emit,

      // --- Theme ---
      setTheme: function (t) { state.theme = t; emit("theme"); },

      // --- Session hydration ---
      hydrate: function (patch) {
        Object.assign(state, patch);
        if (patch.investigation) state.blocks = patch.investigation.blocks || [];
        emit("hydrate");
      },

      // --- Stream control ---
      streamReset: streamReset,
      streamStart: streamStart,
      streamError: streamError,
      streamEvent: streamEvent,

      // --- Conversations ---
      loadConversationIndex: loadConversationIndex,
      loadConversation: loadConversation,
      saveConversation: saveConversation,
      appendToHistory: appendToHistory,
      deleteConversation: function (id) {
        MQ.storage.conversations.remove(id);
        loadConversationIndex();
        if (state.conversation && state.conversation.id === id) {
          state.conversation = null;
          emit("conversation");
        }
      },

      // --- Notebooks ---
      loadNotebookIndex: loadNotebookIndex,
      loadNotebook: loadNotebook,
      saveNotebook: saveNotebook,
      deleteNotebook: function (id) {
        MQ.storage.notebooks.remove(id);
        loadNotebookIndex();
        if (state.notebook && state.notebook.id === id) {
          state.notebook = null;
          emit("notebook");
        }
      },

      // --- Explorer ---
      loadExplorerIndex: loadExplorerIndex,
      loadExplorer: loadExplorer,
      saveExplorer: saveExplorer,
      deleteExplorer: function (id) {
        MQ.storage.explorer.remove(id);
        loadExplorerIndex();
        if (state.explorer && state.explorer.id === id) {
          state.explorer = null;
          emit("explorer");
        }
      },

      // --- Datasets ---
      setDatasets: function (datasets, total) {
        state.datasets = datasets;
        state.datasetsTotal = total;
        emit("datasets");
      },
      setDatasetsLoading: function (v) { state.datasetsLoading = v; emit("datasets"); },
      setDatasetDetail: function (d) { state.datasetDetail = d; emit("datasetDetail"); },
      setDatasetColumns: function (c) { state.datasetColumns = c; emit("datasetColumns"); },
      setDatasetDocuments: function (d) { state.datasetDocuments = d; emit("datasetDocuments"); },

      // --- Stage blocks ---
      setBlocks: function (blocks) { state.blocks = blocks; emit("blocks"); },
      addBlock: function (block, atIndex) {
        if (typeof atIndex === "number") state.blocks.splice(atIndex, 0, block);
        else state.blocks.push(block);
        reindex(state.blocks);
        emit("blocks");
        return block;
      },
      patchBlock: function (id, patch) {
        var b = state.blocks.find(function (x) { return x.id === id; });
        if (b) { Object.assign(b, patch); emit("block:" + id); }
        return b;
      },
      moveBlock: function (id, beforeId) {
        var from = state.blocks.findIndex(function (x) { return x.id === id; });
        if (from < 0) return;
        var block = state.blocks.splice(from, 1)[0];
        var to = beforeId == null
          ? state.blocks.length
          : state.blocks.findIndex(function (x) { return x.id === beforeId; });
        if (to < 0) to = state.blocks.length;
        state.blocks.splice(to, 0, block);
        reindex(state.blocks);
        emit("blocks");
      },
      removeBlock: function (id) {
        state.blocks = state.blocks.filter(function (x) { return x.id !== id; });
        reindex(state.blocks);
        emit("blocks");
      },

      // --- Derived selectors ---
      sourceCount: function () {
        var ids = {};
        state.blocks.forEach(function (b) {
          if (b.type === "query" && b.result && b.result.provenance) ids[b.result.provenance.sourceId] = 1;
          if (b.type === "source" && b.provenance) ids[b.provenance.sourceId] = 1;
        });
        // Also count from stream cards
        state.stream.cards.forEach(function (c) {
          if (c.kind === "datasets_ranked" && c.candidates) {
            c.candidates.forEach(function (d) { ids[d.package_id] = 1; });
          }
        });
        return Math.max(1, Object.keys(ids).length);
      }
    };
  }

  MQ.store = createStore();
})(window);
