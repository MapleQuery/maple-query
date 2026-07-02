/**
 * Client-side PostHog capture helper.
 *
 * Every call site imports this instead of `posthog-js` directly so a
 * missing key silences the call — the server-rendered SSR pass has no
 * PostHog on `window`, and unconfigured local runs never called
 * `posthog.init` so `capture` would throw. Guarding here keeps the call
 * sites free of noise.
 */
import posthog from "posthog-js";

export function track(
  event: string,
  properties?: Record<string, unknown>,
): void {
  if (typeof window === "undefined") return;
  if (!process.env.NEXT_PUBLIC_POSTHOG_KEY) return;
  try {
    posthog.capture(event, properties);
  } catch {
    // PostHog raises if `init` never ran (e.g. StrictMode double-render
    // before the provider mounted). Swallow — analytics must never
    // interrupt a user action.
  }
}
