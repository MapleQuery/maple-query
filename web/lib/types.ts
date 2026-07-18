import { z } from "zod";

// ---------------------------------------------------------------------------
// History (client-supplied conversation transcript for POST /chat).
// Shape mirrors agent-service's request contract.
// ---------------------------------------------------------------------------

export const HistoryMessageSchema = z.union([
  z.object({ role: z.literal("user"), content: z.string() }),
  z.object({
    role: z.literal("assistant"),
    content: z.string().nullable().default(""),
    tool_calls: z.array(z.unknown()).optional(),
  }),
  z.object({
    role: z.literal("tool"),
    tool_call_id: z.string(),
    content: z.string(),
  }),
  z.object({
    role: z.literal("system"),
    content: z.string(),
    mq_summary: z.boolean().optional(),
  }),
]);
export type HistoryMessage = z.infer<typeof HistoryMessageSchema>;

export interface ChatRequest {
  conversation_id: string;
  question: string;
  history: HistoryMessage[];
  /** Client-held turn records echoed back so the server-side memory
   * phase (replay skip-hints, clarify follow-ups) can use them. */
  turn_records?: Record<string, unknown>[];
}

// ---------------------------------------------------------------------------
// SSE event schema. Matches services/semantic-enrich core/agent_events.py.
// Every payload is validated at runtime; a malformed frame is dropped, not
// fatal.
// ---------------------------------------------------------------------------

const RowsShape = z.array(z.record(z.unknown()));

const TurnStart = z.object({
  conversation_id: z.string(),
  turn_id: z.string(),
  cached: z.boolean(),
});

const CacheHit = z.object({ cache_key_prefix: z.string() });

const RetrievalStarted = z.object({
  query: z.string(),
  k: z.number(),
});

const DatasetCandidate = z.object({
  package_id: z.string(),
  title: z.string().nullable().optional(),
  summary: z.string().nullable().optional(),
  distance: z.number().nullable().optional(),
});
const DatasetsRanked = z.object({
  candidates: z.array(DatasetCandidate),
});

const ColumnCandidate = z.object({
  package_id: z.string(),
  column_name: z.string(),
  description: z.string().nullable().optional(),
  distance: z.number().nullable().optional(),
});
const ColumnsRanked = z.object({
  package_ids: z.array(z.string()),
  candidates: z.array(ColumnCandidate),
});

const SampleRows = z.object({
  package_id: z.string(),
  rows: RowsShape,
});

const SqlGenerated = z.object({
  sql: z.string(),
  rationale: z.string(),
});

const SqlGuarded = z.object({
  accepted: z.boolean(),
  reason: z.string().nullable(),
  sql_final: z.string(),
  dry_run_bytes: z.number().nullable(),
});

const SqlExecuted = z.object({
  row_count: z.number(),
  bytes_billed: z.number().nullable(),
  elapsed_ms: z.number().nullable(),
  sample_rows: RowsShape.optional().default([]),
});

const Rows = z.object({
  sql_call_id: z.string().optional(),
  rows: RowsShape,
  is_last: z.boolean(),
});

const MessageDelta = z.object({ delta: z.string() });

const CostUpdate = z.object({
  tokens_in_total: z.number(),
  tokens_out_total: z.number(),
  dollars_spent: z.number(),
});

const BudgetExceeded = z.object({
  which: z.enum(["tool_calls", "sql_executions"]),
  value: z.number(),
  cap: z.number(),
});

const TurnTimeout = z.object({
  elapsed_ms: z.number(),
  cap_ms: z.number(),
});

const ToolError = z.object({
  tool: z.string(),
  message: z.string(),
});

const TurnRecord = z.object({
  record: z.record(z.unknown()),
});

const Done = z.object({
  turn_id: z.string(),
  total_tool_calls: z.number(),
  total_dollars: z.number(),
  elapsed_ms: z.number(),
});

const ErrorEvt = z.object({
  message: z.string(),
  retryable: z.boolean(),
});

export const AgentEventSchemas = {
  turn_start: TurnStart,
  cache_hit: CacheHit,
  retrieval_started: RetrievalStarted,
  datasets_ranked: DatasetsRanked,
  columns_ranked: ColumnsRanked,
  sample_rows: SampleRows,
  sql_generated: SqlGenerated,
  sql_guarded: SqlGuarded,
  sql_executed: SqlExecuted,
  rows: Rows,
  message_delta: MessageDelta,
  cost_update: CostUpdate,
  budget_exceeded: BudgetExceeded,
  turn_timeout: TurnTimeout,
  tool_error: ToolError,
  turn_record: TurnRecord,
  done: Done,
  error: ErrorEvt,
} as const;

export type AgentEventName = keyof typeof AgentEventSchemas;

export type AgentEvent = {
  [K in AgentEventName]: { name: K; payload: z.infer<(typeof AgentEventSchemas)[K]> };
}[AgentEventName];

export type DoneEvent = z.infer<typeof Done>;
export type ErrorEvent = z.infer<typeof ErrorEvt>;
export type DatasetCandidateT = z.infer<typeof DatasetCandidate>;
export type ColumnCandidateT = z.infer<typeof ColumnCandidate>;

// ---------------------------------------------------------------------------
// REST endpoints
// ---------------------------------------------------------------------------

export interface DatasetSummary {
  package_id: string;
  title: string;
  summary: string;
  grain?: string | null;
  measures?: string[] | null;
  dimensions?: string[] | null;
  date_range_start?: string | null;
  date_range_end?: string | null;
  distance?: number | null;
}

export interface DatasetsResponse {
  datasets: DatasetSummary[];
  total: number;
}

export interface ColumnInfo {
  column_name: string;
  semantic_type?: string | null;
  description?: string | null;
  sample_values?: unknown[] | null;
}

export interface ColumnsResponse {
  package_id: string;
  columns: ColumnInfo[];
}

export interface DocumentInfo {
  document_id: string;
  title?: string | null;
  source_url: string;
  file_format: string;
  language: string;
  row_count?: number | null;
  published_date?: string | null;
  is_representative?: boolean;
}

export interface DocumentsResponse {
  package_id: string;
  documents: DocumentInfo[];
}

export type SqlRunStatus =
  | "ok"
  | "guard_rejected"
  | "column_not_in_doc"
  | "budget_exceeded"
  | "execution_error";

export interface SqlRunResponse {
  status: SqlRunStatus;
  reason: string | null;
  sql_final: string;
  rows: Record<string, unknown>[];
  row_count: number;
  bytes_billed: number | null;
  elapsed_ms: number | null;
  truncated?: boolean;
}
