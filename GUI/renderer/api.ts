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
  show_if?: Record<string, unknown>;
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
  pre_views: string[];
  companions: Record<string, string>;
  ppm_views: Record<string, "x" | "customdata">;
  has_table: boolean;
  exportable: boolean;
  blocked: string | null;
  schema: JsonSchema;
  params: Record<string, unknown> | null;
}

export interface TablePage {
  columns: string[];
  rows: unknown[][];
  total: number;
}

export interface GroupStyle {
  color: string;
  symbol: string;
}

export interface PlotStyle {
  groups: Record<string, GroupStyle>;
  marker_size: number;
  marker_opacity: number;
  font_size: number;
  line_width: number;
  show_ellipses: boolean;
}

export type StyleOverride = Partial<Omit<PlotStyle, "groups">> & { groups?: Record<string, Partial<GroupStyle>> };

export interface Annotation {
  id: string;
  ppm: number;
  label: string;
  angle: number;
  ax?: number;
  ay?: number;
}

export interface AnnotationSettings {
  font_size?: number;
  font_family?: string;
  font_weight?: string;
  font_style?: string;
  line_styles: Record<string, { color?: string; dash?: string; width?: number }>;
  targets: { groups: string[]; samples: string[] };
}

export interface StageStyle {
  override: StyleOverride;
  effective: PlotStyle;
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

  figure(id: string, view: string, opts: { ppm?: number; mode?: string } = {}): Promise<any> {
    const at = opts.ppm === undefined ? "" : `&ppm=${opts.ppm}`;
    const mode = opts.mode === undefined ? "" : `&mode=${encodeURIComponent(opts.mode)}`;
    return this.req(`/stages/${id}/figure?view=${encodeURIComponent(view)}${at}${mode}`).then((r) => r.json());
  }

  table(id: string, offset = 0, limit = 100): Promise<TablePage> {
    return this.req(`/stages/${id}/table?offset=${offset}&limit=${limit}`).then((r) => r.json());
  }

  private post(path: string, body?: unknown): Promise<any> {
    return this.req(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) }).then((r) => r.json());
  }

  exportCsv(id: string, path: string): Promise<unknown> {
    return this.post(`/stages/${id}/export`, { path });
  }

  saveProject(path: string): Promise<unknown> {
    return this.post("/session/save", { path });
  }

  loadProject(path: string): Promise<unknown> {
    return this.post("/session/load", { path });
  }

  restore(): Promise<{ restored: boolean }> {
    return this.post("/session/restore");
  }

  private put(path: string, body: unknown): Promise<any> {
    return this.req(path, { method: "PUT", body: JSON.stringify(body) }).then((r) => r.json());
  }

  listAnnotations(): Promise<Annotation[]> {
    return this.req("/annotations").then((r) => r.json());
  }

  addAnnotation(ppm: number, label: string, angle: number): Promise<Annotation> {
    return this.post("/annotations", { ppm, label, angle });
  }

  deleteAnnotation(id: string): Promise<unknown> {
    return this.req(`/annotations/${id}`, { method: "DELETE" }).then((r) => r.json());
  }

  importAnnotations(path: string): Promise<Annotation[]> {
    return this.post("/annotations/import", { path });
  }

  patchAnnotation(id: string, patch: Partial<Annotation>): Promise<Annotation> {
    return this.req(`/annotations/${id}`, { method: "PATCH", body: JSON.stringify(patch) }).then((r) => r.json());
  }

  getAnnotationSettings(): Promise<AnnotationSettings> {
    return this.req("/annotations/settings").then((r) => r.json());
  }

  putAnnotationSettings(settings: AnnotationSettings): Promise<AnnotationSettings> {
    return this.put("/annotations/settings", settings);
  }

  exportHtml(id: string, path: string, view: string, opts: { ppm?: number; mode?: string } = {}): Promise<unknown> {
    return this.post(`/stages/${id}/html`, { path, view, ...opts });
  }

  exportAnnotations(path: string): Promise<unknown> {
    return this.post("/annotations/export", { path });
  }

  getStyle(): Promise<PlotStyle> {
    return this.req("/style").then((r) => r.json());
  }

  putStyle(style: PlotStyle): Promise<PlotStyle> {
    return this.put("/style", style);
  }

  getStageStyle(id: string): Promise<StageStyle> {
    return this.req(`/stages/${id}/style`).then((r) => r.json());
  }

  putStageStyle(id: string, override: StyleOverride): Promise<StageStyle> {
    return this.put(`/stages/${id}/style`, override);
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
