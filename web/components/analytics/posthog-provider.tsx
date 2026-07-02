"use client";

import * as React from "react";
import { usePathname, useSearchParams } from "next/navigation";
import posthog from "posthog-js";
import { PostHogProvider as Provider } from "posthog-js/react";

const KEY = process.env.NEXT_PUBLIC_POSTHOG_KEY;
const HOST = process.env.NEXT_PUBLIC_POSTHOG_HOST ?? "https://us.i.posthog.com";

/**
 * Initialize PostHog once on the client. When no key is set (local dev,
 * preview envs the operator hasn't wired) the provider still renders,
 * but capture calls no-op because `posthog.init` never ran.
 */
function initOnce() {
  if (typeof window === "undefined") return;
  if (!KEY) return;
  if ((posthog as unknown as { __loaded?: boolean }).__loaded) return;
  posthog.init(KEY, {
    api_host: HOST,
    // We fire pageviews manually below so the App Router client
    // navigations don't get missed the way a straight `capture_pageview`
    // on init would.
    capture_pageview: false,
    capture_pageleave: true,
    persistence: "localStorage+cookie",
  });
}

function PageViewCapture() {
  const pathname = usePathname();
  const searchParams = useSearchParams();

  React.useEffect(() => {
    if (!KEY) return;
    const url = pathname + (searchParams?.toString() ? `?${searchParams}` : "");
    posthog.capture("$pageview", { $current_url: url });
  }, [pathname, searchParams]);

  return null;
}

export function PostHogProvider({ children }: { children: React.ReactNode }) {
  React.useEffect(() => {
    initOnce();
  }, []);

  if (!KEY) {
    return <>{children}</>;
  }
  return (
    <Provider client={posthog}>
      <React.Suspense fallback={null}>
        <PageViewCapture />
      </React.Suspense>
      {children}
    </Provider>
  );
}
