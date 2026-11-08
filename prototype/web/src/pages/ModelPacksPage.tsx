import React, { useEffect, useState } from "react";
import {
  Cpu,
  Download,
  Sparkles,
  RefreshCw,
  CheckCircle2,
  Loader2,
  AlertCircle,
} from "lucide-react";
import { api, ModelPack, SamplePreset } from "../api/client";

interface ModelPacksPageProps {
  unlocked: boolean;
  onRequireUnlock: () => void;
  onPacksChanged?: (count: number) => void;
}

export const ModelPacksPage: React.FC<ModelPacksPageProps> = ({
  unlocked,
  onRequireUnlock,
  onPacksChanged,
}) => {
  const [packs, setPacks] = useState<ModelPack[]>([]);
  const [samples, setSamples] = useState<SamplePreset[]>([]);
  const [imageId, setImageId] = useState<string>("sample_coins");
  const [query, setQuery] = useState<string>("coins");
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [message, setMessage] = useState<{
    type: "success" | "error";
    text: string;
  } | null>(null);

  const loadData = () => {
    api
      .getPacks()
      .then((res) => {
        setPacks(res.packs);
        const readyCount = res.packs.filter((p) => p.status === "ready").length;
        onPacksChanged?.(readyCount || res.packs.length);
      })
      .catch(() => {});
  };

  useEffect(() => {
    loadData();
    api
      .getSamples()
      .then((res) => {
        setSamples(res.samples);
        const coinSample =
          res.samples.find((s) => s.id === "sample_coins") || res.samples[0];
        if (coinSample) {
          setImageId(coinSample.id);
          setQuery(coinSample.defaultQuery);
        }
      })
      .catch(() => {});
  }, []);

  // Poll every 4 seconds while a pack is training or a re-train is in flight
  // (ready packs keep status 'ready' while a new artifact is being distilled).
  useEffect(() => {
    const hasTraining = packs.some((p) => p.status === "training" || p.trainingInFlight);
    if (!hasTraining) return;
    const t = setInterval(loadData, 4000);
    return () => clearInterval(t);
  }, [packs]);

  const handleTrainSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!unlocked) {
      onRequireUnlock();
      return;
    }
    if (!imageId || !query.trim()) return;

    setSubmitting(true);
    setMessage(null);
    try {
      const res = await api.trainPack(imageId, query.trim());
      setMessage({
        type: "success",
        text: `Distillation Flywheel dimulai untuk pack "${res.slug}_int8.onnx" (status: ${res.status}). Guru LLM (Gemini 2x2, M5) sedang memberi titik pseudo-label untuk melatih YOLO11n INT8.`,
      });
      loadData();
    } catch (err) {
      const status = (err as { status?: number })?.status;
      if (status === 401) {
        onRequireUnlock();
      } else {
        setMessage({
          type: "error",
          text: err instanceof Error ? err.message : "Gagal melatih pack baru",
        });
      }
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto p-4 md:p-6 space-y-8">
      {/* Header */}
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div
            className="inline-flex items-center gap-2 text-xs font-mono-tech uppercase tracking-wider mb-1"
            style={{ color: "var(--brand-emerald)" }}
          >
            <Cpu className="w-3.5 h-3.5" />
            <span>Method 2 &amp; 3 · Modular On-Device INT8 Engine</span>
          </div>
          <h1 className="text-2xl font-semibold display-head">
            Model Packs (.onnx) & Active Distillation Flywheel
          </h1>
          <p
            className="text-xs mt-1 max-w-2xl leading-relaxed"
            style={{ color: "var(--muted-text)" }}
          >
            Setiap objek yang sering dihitung dikemas menjadi file <code>.onnx</code> INT8 berukuran ~2.89 MB. Aplikasi mengunduh pack sesuai kebutuhan dan menghitung offline: ~0.15 detik untuk foto jarang dan ~0.54 detik (mode 2x2 tiled) untuk foto padat.
          </p>
        </div>

        <button
          type="button"
          onClick={loadData}
          className="inline-flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-medium cursor-pointer"
          style={{
            backgroundColor: "var(--surface-card)",
            border: "1px solid var(--hairline-strong)",
            color: "var(--ink)",
          }}
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>Refresh Catalog</span>
        </button>
      </div>

      {/* Installed Model Packs Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {packs.map((pack) => (
          <div
            key={pack.slug}
            className="rounded-2xl p-5 flex flex-col justify-between space-y-4"
            style={{
              backgroundColor: "var(--surface-card)",
              border: "1px solid var(--hairline-strong)",
            }}
          >
            <div className="space-y-3">
              <div className="flex items-start justify-between gap-2">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-mono-tech text-sm font-bold">
                      {pack.slug}_int8.onnx
                    </span>
                    {pack.status === "ready" && (
                      <span
                        className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-mono-tech font-semibold uppercase"
                        style={{
                          backgroundColor: "rgba(52, 211, 153, 0.16)",
                          color: "var(--brand-emerald)",
                        }}
                      >
                        <CheckCircle2 className="w-3 h-3" />
                        Ready
                      </span>
                    )}
                    {pack.status === "training" && (
                      <span
                        className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-mono-tech font-semibold uppercase"
                        style={{
                          backgroundColor: "rgba(232, 185, 74, 0.2)",
                          color: "var(--brand-ochre)",
                        }}
                      >
                        <Loader2 className="w-3 h-3 animate-spin" />
                        Training...
                      </span>
                    )}
                    {pack.status === "failed" && (
                      <span
                        className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-mono-tech font-semibold uppercase"
                        style={{
                          backgroundColor: "rgba(239, 68, 68, 0.2)",
                          color: "var(--status-danger)",
                        }}
                      >
                        Failed
                      </span>
                    )}
                  </div>
                  <div
                    className="text-xs mt-0.5"
                    style={{ color: "var(--body-text)" }}
                  >
                    {pack.displayName}
                  </div>
                </div>

                <div className="text-right font-mono-tech">
                  <div className="text-sm font-bold" style={{ color: "var(--ink)" }}>
                    {pack.sizeMb > 0 ? `${pack.sizeMb.toFixed(2)} MB` : "—"}
                  </div>
                  <div className="text-[10px]" style={{ color: "var(--muted-text)" }}>
                    INT8 Quantized
                  </div>
                </div>
              </div>

              {/* Aliases */}
              <div className="flex flex-wrap items-center gap-1.5">
                <span
                  className="text-[10px] font-mono-tech uppercase"
                  style={{ color: "var(--muted-text)" }}
                >
                  Matched Queries:
                </span>
                {pack.aliases.map((al) => (
                  <span
                    key={al}
                    className="px-2 py-0.5 rounded-md font-mono-tech text-[11px]"
                    style={{
                      backgroundColor: "var(--surface-strong)",
                      color: "var(--ink)",
                    }}
                  >
                    {al}
                  </span>
                ))}
              </div>
            </div>

            <div
              className="pt-3 flex flex-wrap items-center justify-between gap-2 text-xs font-mono-tech"
              style={{ borderTop: "1px solid var(--hairline)" }}
            >
              <div style={{ color: "var(--muted-text)" }}>
                imgsz={pack.imgsz} · {pack.trainedEpochs} epochs
                {pack.valMapePct != null ? ` · MAPE ${pack.valMapePct}%` : ""}
              </div>

              {pack.status === "ready" && (
                <a
                  href={pack.downloadUrl}
                  download={`${pack.slug}_int8.onnx`}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold cursor-pointer"
                  style={{
                    backgroundColor: "var(--nav-active-bg)",
                    color: "var(--nav-active-fg)",
                  }}
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>Download .onnx</span>
                </a>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Auto-Train New Class Form */}
      <div
        className="rounded-2xl p-6 space-y-4"
        style={{
          backgroundColor: "var(--surface-card)",
          border: "1px solid var(--hairline-strong)",
        }}
      >
        <div className="space-y-1">
          <div className="inline-flex items-center gap-1.5 text-sm font-semibold">
            <Sparkles className="w-4 h-4" style={{ color: "var(--brand-pink)" }} />
            <span>Auto-Train New Class (LLM Teacher &rarr; YOLO11n INT8 .onnx)</span>
          </div>
          <p className="text-xs" style={{ color: "var(--muted-text)" }}>
            Pilih gambar referensi dan nama objek baru. Sistem akan memanggil Gemini 3.8 Flash 2x2 (M5) sebagai Guru Validator, membuat kotak pseudo-label adaptif + augmentasi multi-skala, lalu mengekspor bobot <code>.onnx</code> INT8 siap pakai.
          </p>
        </div>

        <form onSubmit={handleTrainSubmit} className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div>
            <label className="block text-xs font-medium mb-1" style={{ color: "var(--body-text)" }}>
              Reference Sample Image
            </label>
            <select
              value={imageId}
              onChange={(e) => {
                const id = e.target.value;
                setImageId(id);
                const found = samples.find((s) => s.id === id);
                if (found) setQuery(found.defaultQuery);
              }}
              className="w-full px-3 py-2 rounded-xl text-xs font-mono-tech outline-none"
              style={{
                backgroundColor: "var(--surface-composer)",
                border: "1px solid var(--hairline-strong)",
                color: "var(--ink)",
              }}
            >
              {samples.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.filename} (GT: {s.groundTruth})
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-xs font-medium mb-1" style={{ color: "var(--body-text)" }}>
              Target Class Query
            </label>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="mis. coins, blueberry, container..."
              className="w-full px-3 py-2 rounded-xl text-xs font-mono-tech outline-none"
              style={{
                backgroundColor: "var(--surface-composer)",
                border: "1px solid var(--hairline-strong)",
                color: "var(--ink)",
              }}
            />
          </div>

          <div className="flex items-end">
            <button
              type="submit"
              disabled={submitting || !query.trim()}
              className="w-full inline-flex items-center justify-center gap-1.5 px-4 py-2 rounded-xl text-xs font-semibold cursor-pointer disabled:opacity-50"
              style={{
                backgroundColor: "var(--brand-pink)",
                color: "#ffffff",
              }}
            >
              <Cpu className="w-4 h-4" />
              <span>
                {submitting ? "Starting Flywheel..." : "Auto-Train New Class"}
              </span>
            </button>
          </div>
        </form>

        {message && (
          <div
            className="p-3 rounded-xl text-xs flex items-center gap-2"
            style={{
              backgroundColor:
                message.type === "error"
                  ? "rgba(239, 68, 68, 0.14)"
                  : "rgba(52, 211, 153, 0.14)",
              color:
                message.type === "error"
                  ? "var(--status-danger)"
                  : "var(--ink)",
              border:
                message.type === "error"
                  ? "1px solid rgba(239, 68, 68, 0.35)"
                  : "1px solid var(--brand-emerald)",
            }}
          >
            {message.type === "error" ? (
              <AlertCircle className="w-4 h-4 shrink-0" />
            ) : (
              <CheckCircle2
                className="w-4 h-4 shrink-0"
                style={{ color: "var(--brand-emerald)" }}
              />
            )}
            <span>{message.text}</span>
          </div>
        )}
        {packs.some((p) => p.trainingInFlight) && (
          <p className="text-[11px] font-mono-tech" style={{ color: "var(--brand-ochre)" }}>
            Re-train berjalan di background; pack lama tetap melayani M2/M3 sampai
            artefak .onnx baru selesai.
          </p>
        )}
      </div>
    </div>
  );
};
