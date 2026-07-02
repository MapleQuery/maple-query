"use client";

import * as React from "react";
import { useRouter } from "next/navigation";
import { notebooks } from "@/lib/storage";
import { uuid } from "@/lib/utils";
import { PageLoader } from "@/components/ui/maple-loader";

export default function NotebookIndexPage() {
  const router = useRouter();
  React.useEffect(() => {
    const list = notebooks.list();
    const target = list[0]?.id ?? uuid();
    router.replace(`/notebook/${target}`);
  }, [router]);

  return <PageLoader />;
}
