"use client";

import { useParams, useSearchParams } from "next/navigation";
import { ChatContainer } from "@/components/chat/chat-container";

export default function ChatByIdPage() {
  const params = useParams<{ conversationId: string }>();
  const search = useSearchParams();
  const id = params?.conversationId ?? "new";
  const initialQuestion = search?.get("q") ?? undefined;
  return (
    <ChatContainer conversationId={id} initialQuestion={initialQuestion} />
  );
}
