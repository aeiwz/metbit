export type Status = "empty" | "running" | "done" | "failed" | "stale";

export interface FieldSchema {
  type?: string;
  title?: string;
  description?: string;
  default?: unknown;
  enum?: string[];
  minimum?: number;
  maximum?: number;
  format?: string;
  pick_ppm?: boolean;
}

export interface JsonSchema {
  properties: Record<string, FieldSchema>;
  required?: string[];
}

export interface StageInfo {
  id: string;
  title: string;
  section: string;
  requires: string | null;
  status: Status;
  views: string[];
  ppm_views: Record<string, "x" | "customdata">;
  has_table: boolean;
  schema: JsonSchema;
  params: Record<string, unknown> | null;
}

export interface TablePage {
  columns: string[];
  rows: unknown[][];
  total: number;
}

export interface SseEvent {
  event: string;
  data: any;
}

export class ApiError extends Error {
  constructor(public status: number, public detail: unknown) {
    super(typeof detail === "string" ? detail : `HTTP ${status}`);
  }
}

export function parseSse(buffer: string): { events: SseEvent[]; rest: string } {
  const blocks = buffer.split("\n\n");
  const rest = blocks.pop() ?? "";
  const events = blocks.filter(Boolean).map((block) => {
    let event = "message";
    let data = "";
    for (const line of block.split("\n")) {
      if (line.startsWith("event: ")) event = line.slice(7);
      else if (line.startsWith("data: ")) data += line.slice(6);
    }
    return { event, data: data ? JSON.parse(data) : null };
  });
  return { events, rest };
}

export class Api {
  constructor(private base: string, private token: string) {}

  static async connect(): Promise<Api> {
    const { port, token } = await window.metbit.backendInfo();
    return new Api(`http://127.0.0.1:${port}`, token);
  }

  private async req(path: string, init: RequestInit = {}): Promise<Response> {
    const r = await fetch(this.base + path, {
      ...init,
      headers: { Authorization: `Bearer ${this.token}`, "Content-Type": "application/json" },
    });
    if (!r.ok) {
      const body = await r.json().catch(() => ({}));
      throw new ApiError(r.status, body.detail ?? r.statusText);
    }
    return r;
  }

  stages(): Promise<StageInfo[]> {
    return this.req("/stages").then((r) => r.json());
  }

  figure(id: string, view: string): Promise<any> {
    return this.req(`/stages/${id}/figure?view=${encodeURIComponent(view)}`).then((r) => r.json());
  }

  table(id: string, offset = 0, limit = 100): Promise<TablePage> {
    return this.req(`/stages/${id}/table?offset=${offset}&limit=${limit}`).then((r) => r.json());
  }

  async run(id: string, params: Record<string, unknown>, onEvent: (e: SseEvent) => void): Promise<void> {
    const { job_id } = await this.req(`/stages/${id}/run`, {
      method: "POST",
      body: JSON.stringify(params),
    }).then((r) => r.json());
    // fetch, not EventSource: EventSource cannot send the Authorization header
    const r = await this.req(`/jobs/${job_id}/events`);
    const reader = r.body!.pipeThrough(new TextDecoderStream()).getReader();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) return;
      const parsed = parseSse(buf + value);
      buf = parsed.rest;
      parsed.events.forEach(onEvent);
    }
  }
}
