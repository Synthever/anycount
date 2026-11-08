import React, { useEffect, useState, useRef } from "react";
import {
  Sparkles,
  History,
  X,
  CheckCircle2,
  AlertCircle,
  Image as ImageIcon,
  Layers,
  Zap,
} from "lucide-react";
import {
  api,
  RunDetail,
  SamplePreset,
  MethodResult,
} from "../api/client";
import { MethodResultCard } from "../components/MethodResultCard";
import { Composer } from "../components/Composer";

interface PlaygroundPageProps {
  unlocked: boolean;
  onRequireUnlock: () => void;
  onPacksChanged?: () => void;
}

export const PlaygroundPage: React.FC<PlaygroundPageProps> = ({
  unlocked,
  onRequireUnlock,
  onPacksChanged,
}) => {
  const [samples, setSamples] = useState<SamplePreset[]>([]);
  const [selectedImageId, setSelectedImageId] = useState<string>("sample_apples");
  const [selectedImageLabel, setSelectedImageLabel] = useState<string>("Upload gambar");
  const [selectedImageUrl, setSelectedImageUrl] = useState<string>(
    "/api/images/sample_apples/raw"
  );
  const [selectedImageDim, setSelectedImageDim] = useState<{ w: number; h: number }>({
    w: 1000,
    h: 667,
  });
  const [groundTruth, setGroundTruth] = useState<number | null>(289);
  const [query, setQuery] = useState<string>("apples");

  const [compareMode, setCompareMode] = useState<boolean>(true);
  const [selectedMethods, setSelectedMethods] = useState<string[]>(["m2", "m4", "m7"]);

  const [activeRun, setActiveRun] = useState<RunDetail | null>(null);
  const [recentRuns, setRecentRuns] = useState<RunDetail[]>([]);
  const [showHistory, setShowHistory] = useState<boolean>(false);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [historyLoading, setHistoryLoading] = useState<boolean>(false);
  const [streamStalled, setStreamStalled] = useState<boolean>(false);

  const [running, setRunning] = useState<boolean>(false);
  const [uploading, setUploading] = useState<boolean>(false);
  const [trainingPack, setTrainingPack] = useState<boolean>(false);
  const [banner, setBanner] = useState<{
    type: "info" | "error" | "success";
    text: string;
  } | null>(null);

  const unsubRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    api
      .getSamples()
      .then((res) => {
        setSamples(res.samples);
        const def =
          res.samples.find((s) => s.id === "sample_apples") || res.samples[0];
        if (def) {
          setSelectedImageId(def.id);
          // Keep the neutral upload label: only a picked file or a picked
          // sample preset should change what the composer button says.
          setSelectedImageUrl(def.url);
          setSelectedImageDim({ w: def.width, h: def.height });
          setGroundTruth(def.groundTruth);
          setQuery(def.defaultQuery);
        }
      })
      .catch(() => {});

    return () => {
      unsubRef.current?.();
    };
  }, []);

  const refreshHistory = () => {
    if (!unlocked) {
      setRecentRuns([]);
      setHistoryError(null);
      return;
    }
    setHistoryLoading(true);
    api
      .listRuns()
      .then((res) => {
        setRecentRuns(res.runs);
        setHistoryError(null);
      })
      .catch((err) => {
        const status = (err as { status?: number })?.status;
        setRecentRuns([]);
        setHistoryError(
          status === 401
            ? "Sesi terkunci. Masukkan passcode untuk melihat riwayat run."
            : "Gagal memuat riwayat run. Coba lagi sebentar lagi."
        );
      })
      .finally(() => setHistoryLoading(false));
  };

  // Runs (and their point data) are session-gated: never fetch while locked
  // and never auto-open the newest run on mount.
  useEffect(() => {
    refreshHistory();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [unlocked]);

  // A fetched run carries points and a render URL. Logging out must not leave
  // that result on screen for the next viewer of the kiosk.
  useEffect(() => {
    if (!unlocked) {
      unsubRef.current?.();
      unsubRef.current = null;
      setActiveRun(null);
      setRunning(false);
      setStreamStalled(false);
    }
  }, [unlocked]);

  const handleSelectSample = (sample: SamplePreset) => {
    setSelectedImageId(sample.id);
    setSelectedImageLabel(sample.filename.replace(/\.jpg$/i, ""));
    setSelectedImageUrl(sample.url);
    setSelectedImageDim({ w: sample.width, h: sample.height });
    setGroundTruth(sample.groundTruth);
    setQuery(sample.defaultQuery);
    setBanner(null);
  };

  const handleUploadFile = async (file: File) => {
    if (!unlocked) {
      onRequireUnlock();
      return;
    }
    setUploading(true);
    setBanner(null);
    try {
      const uploaded = await api.uploadImage(file);
      setSelectedImageId(uploaded.id);
      setSelectedImageLabel(file.name);
      setSelectedImageUrl(uploaded.url);
      setSelectedImageDim({ w: uploaded.width, h: uploaded.height });
      setGroundTruth(null);
      setBanner({
        type: "success",
        text: `Gambar "${uploaded.filename}" (${uploaded.width}×${uploaded.height}) siap diuji.`,
      });
    } catch (err) {
      const status = (err as { status?: number })?.status;
      if (status === 401) {
        onRequireUnlock();
      } else {
        setBanner({
          type: "error",
          text: err instanceof Error ? err.message : "Gagal mengunggah gambar",
        });
      }
    } finally {
      setUploading(false);
    }
  };

  const startRunWithMethods = async (
    imgId: string,
    q: string,
    methodsToRun: string[]
  ) => {
    if (!unlocked) {
      onRequireUnlock();
      return;
    }
    if (!imgId || !q.trim() || methodsToRun.length === 0) return;

    unsubRef.current?.();
    setRunning(true);
    setStreamStalled(false);
    setBanner(null);

    try {
      const created = await api.createRun(imgId, q.trim(), methodsToRun);
      setActiveRun(created);
      refreshHistory();

      unsubRef.current = api.subscribeRun(
        created.runId,
        (updated) => {
          setActiveRun(updated);
          if (["completed", "partial", "failed"].includes(updated.status)) {
            setRunning(false);
            setStreamStalled(false);
            refreshHistory();
          }
        },
        (stalled) => setStreamStalled(stalled)
      );
    } catch (err) {
      setRunning(false);
      const status = (err as { status?: number })?.status;
      if (status === 401) {
        onRequireUnlock();
      } else {
        setBanner({
          type: "error",
          text: err instanceof Error ? err.message : "Gagal memulai eksekusi",
        });
      }
    }
  };

  const handleSubmitRun = () => {
    startRunWithMethods(selectedImageId, query, selectedMethods);
  };

  const handleRunValidatorM4 = (run: RunDetail) => {
    const nextMethods = Array.from(
      new Set([...run.methods.map((m) => m.methodId), "m4"])
    );
    setCompareMode(nextMethods.length > 1);
    setSelectedMethods(nextMethods);
    startRunWithMethods(run.imageId, run.query, nextMethods);
  };

  const handleAutoTrainPack = async (run: RunDetail) => {
    if (!unlocked) {
      onRequireUnlock();
      return;
    }
    setTrainingPack(true);
    setBanner({
      type: "info",
      text: `Memulai Distillation Flywheel (Guru Gemini 2x2 M5 -> YOLO11n INT8 .onnx) untuk objek "${run.query}"...`,
    });
    try {
      const res = await api.trainPack(run.imageId, run.query);
      setBanner({
        type: "success",
        text: `Job Auto-Train pack "${res.slug}_int8.onnx" sedang berjalan di background! Status pack tampil di kartu Offline ONNX Engine pada sidebar.`,
      });
      onPacksChanged?.();
    } catch (err) {
      const status = (err as { status?: number })?.status;
      if (status === 401) {
        onRequireUnlock();
      } else {
        setBanner({
          type: "error",
          text: err instanceof Error ? err.message : "Gagal memulai Auto-Train",
        });
      }
    } finally {
      setTrainingPack(false);
    }
  };

  const handleRetryMethod = (methodId: MethodResult["methodId"]) => {
    if (!activeRun) return;
    startRunWithMethods(activeRun.imageId, activeRun.query, [methodId]);
  };

  const methodCount = activeRun?.methods.length ?? 0;
  const gridColsClass =
    methodCount <= 1
      ? "grid-cols-1 max-w-4xl mx-auto"
      : methodCount === 2 || methodCount === 4
      ? "grid-cols-1 lg:grid-cols-2"
      : "grid-cols-1 md:grid-cols-2 xl:grid-cols-3";

  return (
    <div className="min-h-[calc(100vh-3.25rem)] flex flex-col justify-between p-4 md:p-6 space-y-6">
      {/* Main Canvas Content */}
      <div className="w-full max-w-7xl mx-auto space-y-5 flex-1">
        {/* Session Context Bar (JRouter Playground Prompt Bubble Style) */}
        <div
          className="rounded-2xl p-4 flex flex-wrap items-center justify-between gap-4"
          style={{
            backgroundColor: "var(--surface-card)",
            border: "1px solid var(--hairline-strong)",
          }}
        >
          <div className="flex items-center gap-3.5 min-w-0">
            <img
              src={activeRun ? activeRun.imageUrl : selectedImageUrl}
              alt={activeRun ? activeRun.query : query}
              className="w-14 h-14 rounded-xl object-cover shrink-0"
              style={{ border: "1px solid var(--hairline-strong)" }}
            />
            <div className="min-w-0 space-y-1">
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-sm font-semibold tracking-tight">
                  Target Query:
                </span>
                <span
                  className="px-2.5 py-0.5 rounded-full text-xs font-mono-tech font-semibold"
                  style={{
                    backgroundColor: "var(--brand-lavender)",
                    color: "#0a0a0a",
                  }}
                >
                  &ldquo;{activeRun ? activeRun.query : query}&rdquo;
                </span>
                {groundTruth != null &&
                  (!activeRun || activeRun.imageId === selectedImageId) && (
                    <span
                      className="px-2 py-0.5 rounded-md font-mono-tech text-[11px]"
                      style={{
                        backgroundColor: "var(--surface-strong)",
                        color: "var(--brand-emerald)",
                      }}
                    >
                      Ref GT: {groundTruth}
                    </span>
                  )}
              </div>
              <div
                className="font-mono-tech text-[11px] flex flex-wrap items-center gap-3"
                style={{ color: "var(--muted-text)" }}
              >
                <span className="inline-flex items-center gap-1">
                  <ImageIcon className="w-3.5 h-3.5" />
                  {activeRun ? activeRun.imageId : selectedImageLabel} (
                  {activeRun ? activeRun.imageWidth : selectedImageDim.w}×
                  {activeRun ? activeRun.imageHeight : selectedImageDim.h})
                </span>
                <span className="inline-flex items-center gap-1">
                  <Layers className="w-3.5 h-3.5" />
                  {activeRun
                    ? `${activeRun.methods.length} metode dibandingkan`
                    : `${selectedMethods.length} metode dipilih`}
                </span>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {streamStalled && running && (
              <span
                className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-[11px] font-mono-tech"
                style={{
                  backgroundColor: "rgba(232, 185, 74, 0.18)",
                  color: "var(--brand-ochre)",
                  border: "1px solid var(--brand-ochre)",
                }}
                title="Koneksi SSE terputus; browser mencoba menyambung ulang otomatis."
              >
                <AlertCircle className="w-3.5 h-3.5" />
                <span>SSE stalled — reconnecting…</span>
              </span>
            )}
            <button
              type="button"
              onClick={handleSubmitRun}
              disabled={running}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-semibold cursor-pointer disabled:opacity-50"
              style={{
                backgroundColor: "var(--brand-pink)",
                color: "#ffffff",
              }}
            >
              <Zap className="w-3.5 h-3.5" />
              <span>{running ? "Running SSE..." : "Run Selected Methods"}</span>
            </button>
          </div>
        </div>

        {/* Status / Notification Banner */}
        {banner && (
          <div
            className="rounded-xl px-4 py-3 text-xs flex items-center justify-between gap-3"
            style={{
              backgroundColor:
                banner.type === "error"
                  ? "rgba(239, 68, 68, 0.14)"
                  : banner.type === "success"
                  ? "rgba(52, 211, 153, 0.14)"
                  : "var(--surface-card)",
              border:
                banner.type === "error"
                  ? "1px solid rgba(239, 68, 68, 0.35)"
                  : banner.type === "success"
                  ? "1px solid var(--brand-emerald)"
                  : "1px solid var(--hairline-strong)",
              color:
                banner.type === "error"
                  ? "var(--status-danger)"
                  : "var(--ink)",
            }}
          >
            <div className="flex items-center gap-2">
              {banner.type === "error" ? (
                <AlertCircle className="w-4 h-4 shrink-0" />
              ) : (
                <CheckCircle2
                  className="w-4 h-4 shrink-0"
                  style={{ color: "var(--brand-emerald)" }}
                />
              )}
              <span>{banner.text}</span>
            </div>
            <button
              type="button"
              onClick={() => setBanner(null)}
              className="p-1 cursor-pointer"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        )}

        {/* Method Result Cards Grid or Ready Preview State */}
        {activeRun ? (
          <div className={`grid gap-4 ${gridColsClass}`}>
            {activeRun.methods.map((m) => (
              <MethodResultCard
                key={`${activeRun.runId}-${m.methodId}`}
                result={m}
                runId={activeRun.runId}
                imageUrl={activeRun.imageUrl}
                imageWidth={activeRun.imageWidth}
                imageHeight={activeRun.imageHeight}
                query={activeRun.query}
                onRunValidatorM4={() => handleRunValidatorM4(activeRun)}
                onAutoTrainPack={() => handleAutoTrainPack(activeRun)}
                onRetryMethod={handleRetryMethod}
                trainingPack={trainingPack}
              />
            ))}
          </div>
        ) : (
          <div
            className="rounded-2xl p-8 text-center space-y-4 max-w-3xl mx-auto"
            style={{
              backgroundColor: "var(--surface-card)",
              border: "1px solid var(--hairline)",
            }}
          >
            <img
              src={selectedImageUrl}
              alt={query}
              className="max-h-72 mx-auto rounded-xl object-contain"
              style={{ border: "1px solid var(--hairline-strong)" }}
            />
            <div className="space-y-1.5">
              <div className="inline-flex items-center gap-1.5 text-sm font-semibold">
                <Sparkles className="w-4 h-4" style={{ color: "var(--brand-pink)" }} />
                <span>Siap Menguji 7 Metode Penghitung Objek AnyCount</span>
              </div>
              <p className="text-xs max-w-xl mx-auto leading-relaxed" style={{ color: "var(--muted-text)" }}>
                Pilih chip sampel di bawah (mis. <code>apples</code>, <code>coins</code>,{" "}
                <code>containers</code>, <code>Heidelbeeren</code>), pilih mode{" "}
                <strong>Single Method</strong> atau <strong>Compare Mode</strong>, lalu tekan{" "}
                <strong>Run Selected Methods</strong> untuk melihat perbandingan hasil dan
                mengoreksi titik langsung di atas kanvas interaktif.
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Recent Runs Slide-Over Drawer */}
      {showHistory && (
        <div
          className="fixed inset-0 z-40 flex justify-end"
          style={{ backgroundColor: "rgba(0, 0, 0, 0.5)" }}
        >
          <div
            className="w-full max-w-md h-full p-5 overflow-y-auto space-y-4"
            style={{
              backgroundColor: "var(--surface-card)",
              borderLeft: "1px solid var(--hairline-strong)",
            }}
          >
            <div className="flex items-center justify-between pb-3"
              style={{ borderBottom: "1px solid var(--hairline)" }}
            >
              <div className="flex items-center gap-2 font-semibold text-sm">
                <History className="w-4 h-4" />
                <span>Recent Runs ({recentRuns.length})</span>
              </div>
              <button
                type="button"
                onClick={() => setShowHistory(false)}
                className="p-1 rounded-lg cursor-pointer"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {!unlocked ? (
              <p className="text-xs leading-relaxed" style={{ color: "var(--muted-text)" }}>
                🔒 Sesi masih terkunci. Masukkan passcode lewat tombol di kanan atas
                untuk melihat riwayat run.
              </p>
            ) : historyError ? (
              <p className="text-xs leading-relaxed" style={{ color: "var(--status-danger)" }}>
                {historyError}
              </p>
            ) : historyLoading ? (
              <p className="text-xs" style={{ color: "var(--muted-text)" }}>
                Memuat riwayat…
              </p>
            ) : recentRuns.length === 0 ? (
              <p className="text-xs" style={{ color: "var(--muted-text)" }}>
                Belum ada riwayat pengujian.
              </p>
            ) : (
              <div className="space-y-2.5">
                {recentRuns.map((r) => (
                  <button
                    key={r.runId}
                    type="button"
                    onClick={() => {
                      setActiveRun(r);
                      setShowHistory(false);
                    }}
                    className="w-full text-left p-3 rounded-xl flex items-center justify-between gap-3 cursor-pointer transition-colors"
                    style={{
                      backgroundColor:
                        activeRun?.runId === r.runId
                          ? "var(--surface-strong)"
                          : "var(--surface-soft)",
                      border: "1px solid var(--hairline)",
                    }}
                  >
                    <div className="min-w-0">
                      <div className="text-xs font-semibold truncate">
                        &ldquo;{r.query}&rdquo; · {r.imageId}
                      </div>
                      <div
                        className="font-mono-tech text-[11px] mt-0.5"
                        style={{ color: "var(--muted-text)" }}
                      >
                        {r.methods.map((m) => `${m.methodId.toUpperCase()}:${m.count ?? m.status}`).join(" | ")}
                      </div>
                    </div>
                    <span
                      className="font-mono-tech text-[10px] uppercase px-2 py-0.5 rounded shrink-0"
                      style={{
                        backgroundColor: "var(--surface-card)",
                        color: "var(--body-text)",
                      }}
                    >
                      {r.status}
                    </span>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Composer (normal flow, no sticky overlay) */}
      <Composer
        samples={samples}
        selectedImageId={selectedImageId}
        selectedImageLabel={selectedImageLabel}
        query={query}
        onQueryChange={setQuery}
        compareMode={compareMode}
        onToggleCompareMode={setCompareMode}
        selectedMethods={selectedMethods}
        onChangeMethods={setSelectedMethods}
        onSelectSample={handleSelectSample}
        onUploadFile={handleUploadFile}
        onSubmitRun={handleSubmitRun}
        onToggleHistory={() => {
          refreshHistory();
          setShowHistory((v) => !v);
        }}
        running={running}
        uploading={uploading}
      />
    </div>
  );
};
