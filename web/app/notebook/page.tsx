"use client";

import * as React from "react";
import { useRouter } from "next/navigation";
import { notebooks } from "@/lib/storage";
import { uuid } from "@/lib/utils";

export default function NotebookIndexPage() {
  const router = useRouter();
  React.useEffect(() => {
    const list = notebooks.list();
    const target = list[0]?.id ?? uuid();
    router.replace(`/notebook/${target}`);
  }, [router]);

  return (
    <div className="grid h-[calc(100vh-4rem)] place-items-center text-sm text-muted">
      Loading notebook…
    </div>
  );
}
