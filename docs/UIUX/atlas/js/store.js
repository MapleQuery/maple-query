/* ============================================================================
   MapleQuery Notebook — store.js
   ----------------------------------------------------------------------------
   A tiny reactive state container. app.js reads/writes state here and
   subscribes to changes; it never mutates the DOM off the back of an API call
   directly. This is the natural home for a future real-time sync layer:
   a CRDT/WebSocket client would apply remote ops through the same mutators
   (addBlock, moveBlock, patchBlock) and every subscribed view would converge.
   ========================================================================== */
(function (global) {
  "use strict";
  var MQ = global.MQ || (global.MQ = {});

  function createStore() {
    var state = {
      session: null,        // Session
      sources: [],          // Source[]
      investigations: [],   // InvestigationSummary[]
      investigation: null,  // Investigation (current)
      blocks: [],           // Block[] (current doc, ordered)
      theme: "light"
    };
    var subs = { "*": [] };

    function emit(evt) {
      (subs[evt] || []).forEach(function (fn) { fn(state); });
      subs["*"].forEach(function (fn) { fn(state, evt); });
    }

    return {
      get: function () { return state; },

      /** Subscribe to an event ("blocks", "investigation", "theme", ...) or "*". */
      on: function (evt, fn) {
        (subs[evt] || (subs[evt] = [])).push(fn);
        return function off() { subs[evt] = subs[evt].filter(function (f) { return f !== fn; }); };
      },

      /* ---- Bulk hydrate from the API ------------------------------------ */
      hydrate: function (patch) {
        Object.assign(state, patch);
        if (patch.investigation) state.blocks = patch.investigation.blocks || [];
        emit("hydrate");
      },

      setTheme: function (t) { state.theme = t; emit("theme"); },

      /* ---- Block mutators (also the remote-op entry points) ------------- */
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

      /* ---- Derived selectors -------------------------------------------- */
      sourceCount: function () {
        // Distinct provenance across ran query blocks + explicit source blocks.
        var ids = {};
        state.blocks.forEach(function (b) {
          if (b.type === "query" && b.result && b.result.provenance) ids[b.result.provenance.sourceId] = 1;
          if (b.type === "source" && b.provenance) ids[b.provenance.sourceId] = 1;
        });
        return Math.max(1, Object.keys(ids).length);
      }
    };

    function reindex(list) { list.forEach(function (b, i) { b.position = i; }); }
  }

  MQ.store = createStore();
})(window);
