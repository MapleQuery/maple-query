"use client";

import { useParams } from "next/navigation";
import { NotebookContainer } from "@/components/notebook/notebook-container";

export default function NotebookByIdPage() {
  const params = useParams<{ notebookId: string }>();
  const id = params?.notebookId ?? "new";
  return <NotebookContainer notebookId={id} />;
}
