"use client";

import * as React from "react";
import { useRouter } from "next/navigation";
import { conversations } from "@/lib/storage";
import { uuid } from "@/lib/utils";
import { PageLoader } from "@/components/ui/maple-loader";

/**
 * /chat with no id redirects to /chat/<most-recent> or spawns a new one.
 * Client-side because the routing decision needs localStorage.
 */
export default function ChatIndexPage() {
  const router = useRouter();

  React.useEffect(() => {
    const index = conversations.list();
    const target = index[0]?.id ?? uuid();
    router.replace(`/chat/${target}`);
  }, [router]);

  return <PageLoader />;
}
