/* ============================================================================
   MapleQuery ATLAS — storage.js
   ----------------------------------------------------------------------------
   localStorage persistence layer. Mirrors production's lib/storage.ts:
   - Namespaced keys (mq:<collection>:v1:<id>)
   - Per-collection LRU indexes (updatedAt-based)
   - Quota-exceeded recovery via oldest-first eviction
   - Typed facades for conversations, notebooks, explorer sessions

   All reads are safe when localStorage is unavailable (returns defaults).
   ========================================================================== */
(function (global) {
  "use strict";
  var MQ = global.MQ || (global.MQ = {});

  var NS = "mq";
  var V = "v1";
  var MAX_ENTRIES = 50;

  function indexKey(c) { return NS + ":" + c + ":" + V + ":index"; }
  function entryKey(c, id) { return NS + ":" + c + ":" + V + ":" + id; }

  function getIndex(c) {
    try {
      var raw = localStorage.getItem(indexKey(c));
      if (!raw) return [];
      var parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed : [];
    } catch (e) { return []; }
  }

  function writeIndex(c, entries) {
    try { localStorage.setItem(indexKey(c), JSON.stringify(entries)); } catch (e) {}
  }

  function loadEntry(c, id) {
    try {
      var raw = localStorage.getItem(entryKey(c, id));
      return raw ? JSON.parse(raw) : null;
    } catch (e) { return null; }
  }

  function isQuotaError(err) {
    return err instanceof DOMException &&
      (err.name === "QuotaExceededError" ||
       err.name === "NS_ERROR_DOM_QUOTA_REACHED" ||
       err.code === 22);
  }

  function attemptWrite(c, id, payload) {
    var key = entryKey(c, id);
    while (true) {
      try {
        localStorage.setItem(key, payload);
        return true;
      } catch (err) {
        if (!isQuotaError(err)) return false;
        var idx = getIndex(c);
        if (idx.length === 0) return false;
        var oldest = idx[idx.length - 1];
        try { localStorage.removeItem(entryKey(c, oldest.id)); } catch (e) {}
        writeIndex(c, idx.slice(0, -1));
      }
    }
  }

  function enforceCap(c, idx) {
    while (idx.length > MAX_ENTRIES) {
      var dropped = idx.pop();
      if (dropped) {
        try { localStorage.removeItem(entryKey(c, dropped.id)); } catch (e) {}
      }
    }
    return idx;
  }

  function saveEntry(c, entry) {
    var payload = JSON.stringify(entry);
    var stored = attemptWrite(c, entry.id, payload);
    if (!stored) return;
    var idx = getIndex(c).filter(function (e) { return e.id !== entry.id; });
    idx.unshift({ id: entry.id, title: entry.title || "", updatedAt: entry.updatedAt });
    idx = enforceCap(c, idx);
    writeIndex(c, idx);
  }

  function deleteEntry(c, id) {
    try { localStorage.removeItem(entryKey(c, id)); } catch (e) {}
    writeIndex(c, getIndex(c).filter(function (e) { return e.id !== id; }));
  }

  // ---------------------------------------------------------------------------
  // Typed facades
  // ---------------------------------------------------------------------------

  MQ.storage = {
    conversations: {
      list: function () { return getIndex("conversations"); },
      load: function (id) { return loadEntry("conversations", id); },
      save: function (c) { return saveEntry("conversations", c); },
      remove: function (id) { return deleteEntry("conversations", id); }
    },
    notebooks: {
      list: function () { return getIndex("notebooks"); },
      load: function (id) { return loadEntry("notebooks", id); },
      save: function (n) { return saveEntry("notebooks", n); },
      remove: function (id) { return deleteEntry("notebooks", id); }
    },
    explorer: {
      list: function () { return getIndex("explorer"); },
      load: function (id) { return loadEntry("explorer", id); },
      save: function (e) { return saveEntry("explorer", e); },
      remove: function (id) { return deleteEntry("explorer", id); }
    }
  };

})(window);
