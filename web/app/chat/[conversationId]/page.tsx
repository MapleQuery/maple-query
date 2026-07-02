"use client";

import { useParams } from "next/navigation";
import { ChatContainer } from "@/components/chat/chat-container";

export default function ChatByIdPage() {
  const params = useParams<{ conversationId: string }>();
  const id = params?.conversationId ?? "new";
  return <ChatContainer conversationId={id} />;
}
