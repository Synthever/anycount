import React, { useRef } from "react";
import {
  Upload,
  ArrowUp,
  History,
  Layers,
  CheckSquare,
  Square,
  Sparkles,
} from "lucide-react";
import { SamplePreset } from "../api/client";

const ALL_METHODS = [
  { id: "m1", label: "M1 SAM3" },
  { id: "m2", label: "M2 ONNX" },
  { id: "m3", label: "M3 ONNX 2x2" },
  { id: "m4", label: "M4 Gemini" },
  { id: "m5", label: "M5 Gemini 2x2" },
  { id: "m6", label: "M6 DeepSeek" },
  { id: "m7", label: "M7 DeepSeek 2x2" },
];

interface ComposerProps {
  samples: SamplePreset[];
  selectedImageId: string;
  selectedImageLabel: string;
  query: string;
  onQueryChange: (q: string) => void;
  compareMode: boolean;
  onToggleCompareMode: (compare: boolean) => void;
  selectedMethods: string[];
  onChangeMethods: (methods: string[]) => void;
  onSelectSample: (sample: SamplePreset) => void;
  onUploadFile: (file: File) => void;
  onSubmitRun: () => void;
  onToggleHistory: () => void;
  running: boolean;
  uploading: boolean;
}

export const Composer: React.FC<ComposerProps> = ({
  samples,
  selectedImageId,
  selectedImageLabel,
  query,
  onQueryChange,
  compareMode,
  onToggleCompareMode,
  selectedMethods,
  onChangeMethods,
  onSelectSample,
  onUploadFile,
  onSubmitRun,
  onToggleHistory,
  running,
  uploading,
}) => {
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleToggleMethod = (id: string) => {
    if (!compareMode) {
      onChangeMethods([id]);
      return;
    }
    if (selectedMethods.includes(id)) {
      if (selectedMethods.length > 1) {
        onChangeMethods(selectedMethods.filter((m) => m !== id));
      }
    } else {
      onChangeMethods([...selectedMethods, id].sort());
    }
  };

  const handlePaste = (e: React.ClipboardEvent) => {
    const items = e.clipboardData?.items;
    if (!items) return;
    for (let i = 0; i < items.length; i++) {
      if (items[i].type.startsWith("image/")) {
        const f = items[i].getAsFile();
        if (f) {
          onUploadFile(f);
          break;
        }
      }
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const f = e.dataTransfer?.files?.[0];
    if (f) {
      onUploadFile(f);
    }
  };

  return (
    <div
      onPaste={handlePaste}
      onDragOver={(e) => e.preventDefault()}
      onDrop={handleDrop}
      className="mx-auto w-full max-w-5xl rounded-2xl p-3.5 shadow-2xl transition-all"
      style={{
        backgroundColor: "var(--surface-composer)",
        border: "1px solid var(--hairline-strong)",
      }}
    >
      {/* Row 1: Quick Sample Chips */}
      <div className="flex items-center gap-1.5 overflow-x-auto scrollbar-hidden pb-2.5 mb-2.5"
        style={{ borderBottom: "1px solid var(--hairline)" }}
      >
        <span
          className="font-mono-tech text-[10px] uppercase tracking-wider px-1.5 shrink-0"
          style={{ color: "var(--muted-text)" }}
        >
          Samples:
        </span>
        {samples.map((s) => {
          const active = s.id === selectedImageId;
          const shortName = s.filename.replace(/\.jpg$/i, "");
          return (
            <button
              key={s.id}
              type="button"
              onClick={() => onSelectSample(s)}
              className="px-2.5 py-1 rounded-full text-[11px] font-mono-tech shrink-0 cursor-pointer transition-colors"
              style={
                active
                  ? {
                      backgroundColor: "var(--nav-active-bg)",
                      color: "var(--nav-active-fg)",
                      fontWeight: 600,
                    }
                  : {
                      backgroundColor: "var(--surface-card)",
                      color: "var(--body-text)",
                      border: "1px solid var(--hairline)",
                    }
              }
            >
              {shortName}
            </button>
          );
        })}
      </div>

      {/* Row 2: Upload + Query Input + Submit */}
      <div className="flex flex-wrap sm:flex-nowrap items-center gap-2.5">
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) onUploadFile(f);
          }}
        />
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          disabled={uploading}
          className="inline-flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-medium shrink-0 cursor-pointer"
          style={{
            backgroundColor: "var(--surface-card)",
            border: "1px solid var(--hairline-strong)",
            color: "var(--ink)",
          }}
        >
          <Upload className="w-3.5 h-3.5" style={{ color: "var(--brand-lavender)" }} />
          <span className="max-w-[140px] truncate">
            {uploading ? "Uploading..." : selectedImageLabel || "Upload Foto"}
          </span>
        </button>

        <input
          type="text"
          value={query}
          onChange={(e) => onQueryChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              onSubmitRun();
            }
          }}
          placeholder="Ketik objek yang ingin dihitung (mis. apples, coins, container, blueberry)..."
          className="flex-1 min-w-[200px] px-3.5 py-2 rounded-xl text-sm outline-none"
          style={{
            backgroundColor: "var(--surface-card)",
            border: "1px solid var(--hairline)",
            color: "var(--ink)",
          }}
        />

        <button
          type="button"
          onClick={onToggleHistory}
          title="Riwayat pengujian (Recent Runs)"
          className="inline-flex items-center gap-1 px-2.5 py-2 rounded-xl text-xs font-medium cursor-pointer shrink-0"
          style={{
            backgroundColor: "var(--surface-card)",
            border: "1px solid var(--hairline)",
            color: "var(--body-text)",
          }}
        >
          <History className="w-3.5 h-3.5" />
          <span className="hidden md:inline">History</span>
        </button>

        <button
          type="button"
          onClick={onSubmitRun}
          disabled={running || !selectedImageId || !query.trim()}
          aria-label="Run counting methods"
          className="w-9 h-9 rounded-full flex items-center justify-center cursor-pointer shrink-0 disabled:opacity-40 transition-transform active:scale-95"
          style={{
            backgroundColor: "var(--nav-active-bg)",
            color: "var(--nav-active-fg)",
          }}
        >
          <ArrowUp className="w-4 h-4" />
        </button>
      </div>

      {/* Row 3: Single vs Compare Mode & Method Selector */}
      <div className="mt-2.5 pt-2 flex flex-wrap items-center justify-between gap-2 text-xs"
        style={{ borderTop: "1px solid var(--hairline)" }}
      >
        <div className="flex flex-wrap items-center gap-1.5">
          {/* Mode Switcher */}
          <div
            className="inline-flex rounded-lg p-0.5 mr-1"
            style={{
              backgroundColor: "var(--surface-card)",
              border: "1px solid var(--hairline)",
            }}
          >
            <button
              type="button"
              onClick={() => {
                onToggleCompareMode(false);
                onChangeMethods([selectedMethods[0] || "m4"]);
              }}
              className="px-2 py-1 rounded-md text-[11px] font-medium cursor-pointer"
              style={
                !compareMode
                  ? {
                      backgroundColor: "var(--nav-active-bg)",
                      color: "var(--nav-active-fg)",
                    }
                  : { color: "var(--muted-text)" }
              }
            >
              Single Method
            </button>
            <button
              type="button"
              onClick={() => {
                onToggleCompareMode(true);
                if (selectedMethods.length < 2) {
                  onChangeMethods(["m2", "m4", "m7"]);
                }
              }}
              className="inline-flex items-center gap-1 px-2 py-1 rounded-md text-[11px] font-medium cursor-pointer"
              style={
                compareMode
                  ? {
                      backgroundColor: "var(--nav-active-bg)",
                      color: "var(--nav-active-fg)",
                    }
                  : { color: "var(--muted-text)" }
              }
            >
              <Layers className="w-3 h-3" />
              <span>Compare Mode</span>
            </button>
          </div>

          {/* Method Pills */}
          {ALL_METHODS.map((m) => {
            const active = selectedMethods.includes(m.id);
            return (
              <button
                key={m.id}
                type="button"
                onClick={() => handleToggleMethod(m.id)}
                className="inline-flex items-center gap-1 px-2 py-1 rounded-lg font-mono-tech text-[11px] cursor-pointer transition-colors"
                style={
                  active
                    ? {
                        backgroundColor: "var(--brand-lavender)",
                        color: "#0a0a0a",
                        fontWeight: 600,
                      }
                    : {
                        backgroundColor: "var(--surface-card)",
                        color: "var(--body-text)",
                        border: "1px solid var(--hairline)",
                      }
                }
              >
                {compareMode &&
                  (active ? (
                    <CheckSquare className="w-3 h-3" />
                  ) : (
                    <Square className="w-3 h-3" />
                  ))}
                <span>{m.label}</span>
              </button>
            );
          })}
        </div>

        {/* Compare Presets */}
        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={() => {
              onToggleCompareMode(true);
              onChangeMethods(["m2", "m4", "m7"]);
            }}
            className="inline-flex items-center gap-1 px-2 py-1 rounded-lg text-[11px] font-medium cursor-pointer"
            style={{
              backgroundColor: "var(--surface-card)",
              border: "1px solid var(--hairline-strong)",
              color: "var(--brand-emerald)",
            }}
          >
            <Sparkles className="w-3 h-3" />
            <span>Fast Compare (M2+M4+M7)</span>
          </button>

          <button
            type="button"
            onClick={() => {
              onToggleCompareMode(true);
              onChangeMethods(["m1", "m2", "m3", "m4", "m5", "m6", "m7"]);
            }}
            className="px-2 py-1 rounded-lg text-[11px] font-medium cursor-pointer"
            style={{
              backgroundColor: "var(--surface-card)",
              border: "1px solid var(--hairline-strong)",
              color: "var(--body-text)",
            }}
          >
            All 7 Methods
          </button>
        </div>
      </div>
    </div>
  );
};
