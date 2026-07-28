/* ============================================================================
   MapleQuery ATLAS — config.js
   ----------------------------------------------------------------------------
   Centralized configuration. Reads from URL params or localStorage so the
   same static build works against local dev, staging, and production.

   To go live: set ?api=https://your-cloud-run-url in the URL bar,
   or call MQ.config.set({ baseUrl, token }) from the console / auth flow.
   ========================================================================== */
(function (global) {
  "use strict";
  var MQ = global.MQ || (global.MQ = {});

  var params = new URLSearchParams(global.location ? global.location.search : "");

  // Persisted config (auth token survives page reloads)
  var stored = {};
  try { stored = JSON.parse(localStorage.getItem("mq-config") || "{}"); } catch (e) {}

  var config = {
    // API base URL — no trailing slash. Cloud Run endpoint or localhost.
    baseUrl: params.get("api") || stored.baseUrl || "https://agent-service-f2ihvaiw6q-uc.a.run.app",

    // Shared bearer token. Set by the auth flow or via ?token= param.
    // Set via ?token= URL param, MQ.config.set({token}), or the auth flow.
    // NEVER hardcode a real token here — it must not be committed.
    token: params.get("token") || stored.token || null,

    // Mode: "live" talks to the real API; "mock" uses local fixtures.
    // Set ?mode=mock in the URL for offline design work.
    mode: params.get("mode") || stored.mode || "live",

    // PostHog (optional, no-ops when absent)
    posthogKey: null,

    /** Merge new config values and persist to localStorage. */
    set: function (patch) {
      Object.keys(patch).forEach(function (k) { config[k] = patch[k]; });
      try {
        localStorage.setItem("mq-config", JSON.stringify({
          baseUrl: config.baseUrl,
          token: config.token,
          mode: config.mode
        }));
      } catch (e) {}
    },

    /** Build Authorization header dict. */
    authHeaders: function () {
      var h = {};
      if (config.token) h["Authorization"] = "Bearer " + config.token;
      return h;
    }
  };

  MQ.config = config;
})(window);
