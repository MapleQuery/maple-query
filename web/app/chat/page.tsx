"use client";

import * as React from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { conversations } from "@/lib/storage";
import { uuid } from "@/lib/utils";
import { PageLoader } from "@/components/ui/maple-loader";

/**
 * /chat with no id redirects to /chat/<most-recent> or spawns a new one.
 * Client-side because the routing decision needs localStorage. A `?q=`
 * arriving from the landing page forces a fresh conversation so it
 * doesn't jam a pre-filled question into an existing thread.
 */
export default function ChatIndexPage() {
  const router = useRouter();
  const search = useSearchParams();

  React.useEffect(() => {
    const q = search?.get("q");
    const index = conversations.list();
    const target = q ? uuid() : (index[0]?.id ?? uuid());
    const qs = q ? `?q=${encodeURIComponent(q)}` : "";
    router.replace(`/chat/${target}${qs}`);
  }, [router, search]);

  return <PageLoader />;
}
