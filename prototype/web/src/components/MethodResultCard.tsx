import React, { useEffect, useState } from "react";
import {
  Edit3,
  Grid,
  Code2,
  Download,
  RefreshCw,
  Sparkles,
  Cpu,
  AlertTriangle,
  Loader2,
  Clock,
} from "lucide-react";
import { MethodResult } from "../api/client";
import { InteractiveCanvas } from "./InteractiveCanvas";

export const METHOD_CATALOG: Record<
  MethodResult["methodId"],
  { code: string; name: string; short: string; accent: string }
> = {
  m1: {
    code: "M1",
    name: "Count-Anything (SAM3 3.4 GB)",
    short: "SAM3 Heavy Baseline",
    accent: "var(--brand-ochre)",
  },
  m2: {
    code: "M2",
    name: "Offline ONNX Pack + LLM Validator",
    short: "2.89 MB INT8 On-Device",
    accent: "var(--brand-mint)",
  },
  m3: {
    code: "M3",
    name: "Gemini 3.8 Flash Single-Pass",
    short: "Best Instance Grounding",
    accent: "var(--brand-pink)",
  },
  m4: {
    code: "M4",
    name: "Gemini 3.8 Flash 2x2 Tiled + NMS",
    short: "High-Detail Dense 2x2",
    accent: "var(--brand-lavender)",
  },
  m5: {
    code: "M5",
    name: "DeepSeek V4.1 Flash Single-Pass",
    short: "Fast Count-Then-Locate",
    accent: "var(--brand-peach)",
  },
  m6: {
    code: "M6",
    name: "DeepSeek V4.1 Flash 2x2 Tiled + NMS",
    short: "Fastest Dense Cloud",
    accent: "var(--brand-emerald)",
  },
};

interface MethodResultCardProps {
  result: MethodResult;
  imageUrl: string;
  imageWidth: number;
  imageHeight: number;
  query: string;
  onRunValidatorM3?: () => void;
  onAutoTrainPack?: () => void;
  onRetryMethod?: (methodId: MethodResult["methodId"]) => void;
  trainingPack?: boolean;
}

export const MethodResultCard: React.FC<MethodResultCardProps> = ({
  result,
  imageUrl,
  imageWidth,
  imageHeight,
  query,
  onRunValidatorM3,
  onAutoTrainPack,
  onRetryMethod,
  trainingPack = false,
}) => {
  const [editMode, setEditMode] = useState(false);
  const [showGrid, setShowGrid] = useState(false);
  const [showRaw, setShowRaw] = useState(false);
  // Manual dot edits are tracked as a delta only; the server values below stay
  // the source of truth so SSE transitions (queued -> completed) render live.
  const [manualDelta, setManualDelta] = useState(0);

  useEffect(() => {
    setManualDelta(0);
  }, [result.status, result.count, result.points.length]);

  const info = METHOD_CATALOG[result.methodId] ?? {
    code: result.methodId.toUpperCase(),
    name: result.methodId,
    short: "",
    accent: "var(--brand-lavender)",
  };

  const elapsedSec =
    result.elapsedMs != null ? `${(result.elapsedMs / 1000).toFixed(2)}s` : "—";

  const hasGridBoxes = Boolean(result.gridBoxes && result.gridBoxes.length > 0);
  // Live count = server truth for the current render + manual dot edits only.
  const serverCount = result.count ?? result.points.length;
  const effectiveCount =
    result.status === "completed" ? Math.max(0, serverCount + manualDelta) : serverCount;

  return (
    <div
      className="rounded-2xl overflow-hidden flex flex-col justify-between transition-all"
      style={{
        backgroundColor: "var(--surface-card)",
        border: "1px solid var(--hairline-strong)",
      }}
    >
      {/* Card Header */}
      <div
        className="px-4 py-3 flex flex-wrap items-center justify-between gap-2"
        style={{ borderBottom: "1px solid var(--hairline)" }}
      >
        <div className="flex items-center gap-2.5 min-w-0">
          <span
            className="font-mono-tech text-xs font-bold px-2 py-0.5 rounded-md shrink-0"
            style={{
              backgroundColor: info.accent,
              color: "#0a0a0a",
            }}
          >
            {info.code}
          </span>
          <div className="min-w-0">
            <div className="text-xs font-semibold truncate" style={{ color: "var(--ink)" }}>
              {info.name}
            </div>
            <div className="text-[11px] truncate" style={{ color: "var(--muted-text)" }}>
              {info.short}
            </div>
          </div>
        </div>

        {/* Status Pill + Action Buttons */}
        <div className="flex flex-wrap items-center gap-1.5">
          {result.status === "queued" && (
            <span
              className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-mono-tech"
              style={{
                backgroundColor: "var(--surface-strong)",
                color: "var(--brand-ochre)",
              }}
            >
              <Clock className="w-3 h-3" />
              <span>Queued</span>
            </span>
          )}
          {result.status === "running" && (
            <span
              className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-mono-tech"
              style={{
                backgroundColor: "var(--surface-strong)",
                color: "var(--brand-lavender)",
              }}
            >
              <Loader2 className="w-3 h-3 animate-spin" />
              <span>Running...</span>
            </span>
          )}
          {result.status === "cold_start" && (
            <span
              className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-mono-tech font-medium"
              style={{
                backgroundColor: "rgba(232, 185, 74, 0.18)",
                color: "var(--brand-ochre)",
              }}
            >
              <span>Cold-Start</span>
            </span>
          )}
          {result.status === "completed" && (
            <span
              className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-mono-tech font-medium"
              style={{
                backgroundColor: "var(--surface-strong)",
                color: "var(--brand-emerald)",
              }}
            >
              <span>{elapsedSec}</span>
            </span>
          )}
          {result.status === "failed" && (
            <span
              className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-mono-tech font-medium"
              style={{
                backgroundColor: "rgba(239, 68, 68, 0.18)",
                color: "var(--status-danger)",
              }}
            >
              <span>Failed</span>
            </span>
          )}

          {result.status === "completed" && (
            <>
              <button
                type="button"
                onClick={() => setEditMode((v) => !v)}
                className="inline-flex items-center gap-1 px-2 py-1 rounded-lg text-[11px] font-medium cursor-pointer transition-colors"
                style={
                  editMode
                    ? {
                        backgroundColor: "var(--brand-pink)",
                        color: "#ffffff",
                      }
                    : {
                        backgroundColor: "var(--surface-strong)",
                        color: "var(--body-text)",
                      }
                }
              >
                <Edit3 className="w-3 h-3" />
                <span>Edit Dots</span>
              </button>

              {hasGridBoxes && (
                <button
                  type="button"
                  onClick={() => setShowGrid((v) => !v)}
                  className="inline-flex items-center gap-1 px-2 py-1 rounded-lg text-[11px] font-medium cursor-pointer transition-colors"
                  style={
                    showGrid
                      ? {
                          backgroundColor: "var(--brand-lavender)",
                          color: "#0a0a0a",
                        }
                      : {
                          backgroundColor: "var(--surface-strong)",
                          color: "var(--body-text)",
                        }
                  }
                >
                  <Grid className="w-3 h-3" />
                  <span>2x2 Grid</span>
                </button>
              )}

              <a
                href={result.renderedUrl}
                download={`anycount_${result.methodId}_${query}.jpg`}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1 px-2 py-1 rounded-lg text-[11px] font-medium cursor-pointer"
                style={{
                  backgroundColor: "var(--surface-strong)",
                  color: "var(--body-text)",
                }}
              >
                <Download className="w-3 h-3" />
                <span>Download JPG</span>
              </a>
            </>
          )}

          <button
            type="button"
            onClick={() => setShowRaw((v) => !v)}
            className="inline-flex items-center gap-1 px-2 py-1 rounded-lg text-[11px] font-mono-tech cursor-pointer"
            style={{
              backgroundColor: showRaw ? "var(--nav-active-bg)" : "var(--surface-strong)",
              color: showRaw ? "var(--nav-active-fg)" : "var(--body-text)",
            }}
          >
            <Code2 className="w-3 h-3" />
            <span>&lt;&gt; Raw</span>
          </button>

          {onRetryMethod && (result.status === "failed" || result.status === "completed") && (
            <button
              type="button"
              onClick={() => onRetryMethod(result.methodId)}
              title="Jalankan ulang metode ini"
              className="p-1 rounded-lg cursor-pointer"
              style={{
                backgroundColor: "var(--surface-strong)",
                color: "var(--muted-text)",
              }}
            >
              <RefreshCw className="w-3 h-3" />
            </button>
          )}
        </div>
      </div>

      {/* Card Body */}
      <div className="p-4 flex-1 flex flex-col justify-center">
        {(result.status === "queued" || result.status === "running") && (
          <div
            className="rounded-xl p-8 flex flex-col items-center justify-center text-center space-y-2.5 min-h-[240px]"
            style={{
              backgroundColor: "var(--surface-soft)",
              border: "1px dashed var(--hairline-strong)",
            }}
          >
            <Loader2
              className="w-7 h-7 animate-spin"
              style={{ color: info.accent }}
            />
            <div className="text-xs font-medium" style={{ color: "var(--ink)" }}>
              {result.status === "queued"
                ? "Menunggu antrean Semaphore(1) untuk menjaga memori VPS..."
                : `Menjalankan ${info.name} pada objek "${query}"...`}
            </div>
            <div className="text-[11px]" style={{ color: "var(--muted-text)" }}>
              Hasil metode ini akan langsung dirender via SSE begitu selesai.
            </div>
          </div>
        )}

        {result.status === "cold_start" && (
          <div
            className="rounded-xl p-5 space-y-4"
            style={{
              backgroundColor: "var(--surface-soft)",
              border: "1px solid var(--brand-ochre)",
            }}
          >
            <div className="flex items-start gap-3">
              <div
                className="w-9 h-9 rounded-xl flex items-center justify-center shrink-0 font-mono-tech font-bold text-xs"
                style={{
                  backgroundColor: "var(--brand-ochre)",
                  color: "#0a0a0a",
                }}
              >
                M2
              </div>
              <div className="space-y-1">
                <div className="text-sm font-semibold" style={{ color: "var(--ink)" }}>
                  Cold-Start: Pack .onnx &ldquo;{query}&rdquo; Belum Tersedia
                </div>
                <p className="text-xs leading-relaxed" style={{ color: "var(--body-text)" }}>
                  {String(
                    result.meta?.message ||
                      `Model pack .onnx untuk '${query}' belum dilatih di katalog offline.`
                  )}{" "}
                  Pada arsitektur Metode 2, objek baru ditangani oleh{" "}
                  <strong>LLM Validator (M3)</strong> terlebih dahulu atau langsung memicu{" "}
                  <strong>Distillation Flywheel</strong> untuk melatih pack INT8{" "}
                  <code>.onnx</code> (~2.89 MB) secara otomatis.
                </p>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2.5 pt-1">
              <button
                type="button"
                onClick={onRunValidatorM3}
                className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-semibold cursor-pointer"
                style={{
                  backgroundColor: "var(--brand-pink)",
                  color: "#ffffff",
                }}
              >
                <Sparkles className="w-3.5 h-3.5" />
                <span>Run LLM Validator (M3)</span>
              </button>

              <button
                type="button"
                onClick={onAutoTrainPack}
                disabled={trainingPack}
                className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-semibold cursor-pointer disabled:opacity-50"
                style={{
                  backgroundColor: "var(--nav-active-bg)",
                  color: "var(--nav-active-fg)",
                }}
              >
                <Cpu className="w-3.5 h-3.5" />
                <span>
                  {trainingPack
                    ? "Auto-Training .onnx Pack..."
                    : "Auto-Train .onnx Pack"}
                </span>
              </button>
            </div>
          </div>
        )}

        {result.status === "failed" && (
          <div
            className="rounded-xl p-5 flex items-start gap-3"
            style={{
              backgroundColor: "rgba(239, 68, 68, 0.1)",
              border: "1px solid rgba(239, 68, 68, 0.35)",
              color: "var(--status-danger)",
            }}
          >
            <AlertTriangle className="w-5 h-5 shrink-0 mt-0.5" />
            <div className="space-y-1 text-xs">
              <div className="font-semibold">Eksekusi Metode Gagal</div>
              <div className="font-mono-tech text-[11px] break-all">
                {String(result.meta?.error || "Terjadi kesalahan saat memproses gambar.")}
              </div>
            </div>
          </div>
        )}

        {result.status === "completed" && (
          <div className="space-y-2.5">
            <InteractiveCanvas
              imageUrl={imageUrl}
              imageWidth={imageWidth}
              imageHeight={imageHeight}
              query={query}
              initialPoints={result.points}
              gridBoxes={result.gridBoxes}
              editMode={editMode}
              showGrid={showGrid}
              methodId={result.methodId}
              onPointsChange={(_pts, delta) => {
                setManualDelta(delta);
              }}
            />

            {result.methodId === "m2" && onRunValidatorM3 && (
              <div className="flex items-center justify-between px-3 py-2 rounded-xl text-xs"
                style={{
                  backgroundColor: "var(--surface-soft)",
                  border: "1px solid var(--hairline)",
                }}
              >
                <span style={{ color: "var(--body-text)" }}>
                  Pack aktif: <code className="font-mono-tech">{String(result.meta?.packSlug || "apple")}</code> ({String(result.meta?.mode || "single_640")})
                </span>
                <button
                  type="button"
                  onClick={onRunValidatorM3}
                  className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-[11px] font-semibold cursor-pointer"
                  style={{
                    backgroundColor: "var(--surface-strong)",
                    color: "var(--ink)",
                  }}
                >
                  <Sparkles className="w-3 h-3" style={{ color: "var(--brand-pink)" }} />
                  <span>Validate with LLM (M3)</span>
                </button>
              </div>
            )}
          </div>
        )}

        {/* Raw JSON Drawer */}
        {showRaw && (
          <pre
            className="mt-3 p-3 rounded-xl font-mono-tech text-[11px] overflow-x-auto max-h-56"
            style={{
              backgroundColor: "var(--surface-soft)",
              border: "1px solid var(--hairline)",
              color: "var(--body-text)",
            }}
          >
            {JSON.stringify(
              {
                methodId: result.methodId,
                status: result.status,
                count: effectiveCount,
                manualDelta,
                elapsedMs: result.elapsedMs,
                promptTokens: result.promptTokens,
                completionTokens: result.completionTokens,
                modelSizeMb: result.modelSizeMb,
                meta: result.meta,
                samplePoints: result.points.slice(0, 12),
              },
              null,
              2
            )}
          </pre>
        )}
      </div>

      {/* Monospace Telemetry Footer */}
      <div
        className="px-4 py-2.5 font-mono-tech text-[11px] flex flex-wrap items-center justify-between gap-2"
        style={{
          backgroundColor: "var(--surface-soft)",
          borderTop: "1px solid var(--hairline)",
          color: "var(--muted-text)",
        }}
      >
        <span>
          {elapsedSec} · {effectiveCount} objects
          {manualDelta !== 0
            ? ` (${manualDelta > 0 ? `+${manualDelta}` : manualDelta} manual)`
            : ""}{" "}
          · {(result.promptTokens ?? 0).toLocaleString()} in ·{" "}
          {(result.completionTokens ?? 0).toLocaleString()} out ·{" "}
          {result.modelSizeMb != null ? `${result.modelSizeMb} MB local` : "0 MB local"}
        </span>
        <span className="uppercase">{result.status}</span>
      </div>
    </div>
  );
};
