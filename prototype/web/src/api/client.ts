export interface Point2D {
  x: number;
  y: number;
}

export interface MethodResult {
  methodId: "m1" | "m2" | "m3" | "m4" | "m5" | "m6" | "m7";
  status: "queued" | "running" | "completed" | "cold_start" | "failed";
  count: number | null;
  points: Point2D[];
  gridBoxes: [number, number, number, number][] | null;
  elapsedMs: number | null;
  promptTokens: number | null;
  completionTokens: number | null;
  modelSizeMb: number | null;
  meta: Record<string, unknown>;
  renderedUrl: string;
}

export interface RunDetail {
  runId: string;
  imageId: string;
  imageUrl: string;
  imageWidth: number;
  imageHeight: number;
  query: string;
  status: string;
  createdAt: string;
  inverted: boolean;
  methods: MethodResult[];
}

export interface SamplePreset {
  id: string;
  filename: string;
  defaultQuery: string;
  groundTruth: number;
  width: number;
  height: number;
  url: string;
}

export interface MontageItem {
  id: string;
  title: string;
  url: string;
  best: string;
}

export interface ModelPack {
  slug: string;
  displayName: string;
  aliases: string[];
  sizeMb: number;
  imgsz: number;
  trainedEpochs: number;
  valMapePct: number | null;
  status: "ready" | "training" | "failed";
  trainingInFlight?: boolean;
  updatedAt: string;
  downloadUrl: string;
}

export interface MethodDocItem {
  id: string;
  code: string;
  name: string;
  badge: string;
  color: string;
  speed: string;
  footprint: string;
  summary: string;
  bestFor: string;
}

export interface GlossaryItem {
  term: string;
  plain: string;
}

export interface DocsPayload {
  methods: MethodDocItem[];
  glossary: GlossaryItem[];
}

async function reqJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const msg = data?.error?.message || `HTTP ${res.status}`;
    const err = new Error(msg) as Error & { code?: string; status?: number };
    err.code = data?.error?.code;
    err.status = res.status;
    throw err;
  }
  return data as T;
}

export const api = {
  getAuthStatus: () => reqJson<{ unlocked: boolean }>("/api/auth/status"),
  unlock: (passcode: string) =>
    reqJson<{ unlocked: boolean }>("/api/auth/unlock", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ passcode }),
    }),
  logout: () => reqJson<{ unlocked: boolean }>("/api/auth/logout", { method: "POST" }),
  uploadImage: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return reqJson<{ id: string; filename: string; width: number; height: number; url: string }>(
      "/api/images",
      { method: "POST", body: fd }
    );
  },
  getSamples: () => reqJson<{ samples: SamplePreset[]; montages: MontageItem[] }>("/api/samples"),
  getPacks: () => reqJson<{ packs: ModelPack[] }>("/api/packs"),
  trainPack: (imageId: string, query: string) =>
    reqJson<{ slug: string; status: string; packStatusUnchanged?: boolean }>("/api/packs/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ imageId, query }),
    }),
  getDocs: () => reqJson<DocsPayload>("/api/docs"),
  createRun: (imageId: string, query: string, methods: string[], invert = false) =>
    reqJson<RunDetail>("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ imageId, query, methods, invert }),
    }),
  listRuns: () => reqJson<{ runs: RunDetail[] }>("/api/runs"),
  getRun: (runId: string) => reqJson<RunDetail>(`/api/runs/${runId}`),
  subscribeRun: (
    runId: string,
    onUpdate: (run: RunDetail) => void,
    onStalled?: (stalled: boolean) => void
  ): (() => void) => {
    const es = new EventSource(`/api/runs/${runId}/stream`);
    es.onopen = () => onStalled?.(false);
    es.onmessage = (ev) => {
      try {
        const parsed = JSON.parse(ev.data) as RunDetail;
        onUpdate(parsed);
        onStalled?.(false);
        if (["completed", "partial", "failed"].includes(parsed.status)) {
          es.close();
        }
      } catch {
        // ignore malformed keepalive
      }
    };
    // A transient drop must not silently kill live updates: EventSource
    // reconnects on its own, so only surface a stalled state.
    es.onerror = () => {
      if (es.readyState !== EventSource.CLOSED) {
        onStalled?.(true);
      }
    };
    return () => es.close();
  },
};
