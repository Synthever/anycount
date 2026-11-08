import React, { useState } from "react";
import { Lock, X, ShieldCheck, AlertCircle } from "lucide-react";
import { api } from "../api/client";

interface PasscodeModalProps {
  open: boolean;
  onClose: () => void;
  onUnlocked: () => void;
}

export const PasscodeModal: React.FC<PasscodeModalProps> = ({
  open,
  onClose,
  onUnlocked,
}) => {
  const [passcode, setPasscode] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!open) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!passcode.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.unlock(passcode.trim());
      if (res.unlocked) {
        setPasscode("");
        onUnlocked();
        onClose();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Kode akses tidak valid");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ backgroundColor: "rgba(0, 0, 0, 0.65)", backdropFilter: "blur(4px)" }}
    >
      <div
        className="w-full max-w-md rounded-2xl p-6 shadow-2xl relative"
        style={{
          backgroundColor: "var(--surface-card)",
          border: "1px solid var(--hairline-strong)",
          color: "var(--ink)",
        }}
      >
        <button
          type="button"
          onClick={onClose}
          aria-label="Close modal"
          className="absolute top-4 right-4 p-1.5 rounded-lg cursor-pointer"
          style={{ color: "var(--muted-text)" }}
        >
          <X className="w-4 h-4" />
        </button>

        <div className="flex items-center gap-3 mb-3">
          <div
            className="w-10 h-10 rounded-xl flex items-center justify-center"
            style={{
              backgroundColor: "var(--surface-strong)",
              color: "var(--brand-lavender)",
            }}
          >
            <Lock className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-base font-semibold tracking-tight">
              Unlock Live Inference Session
            </h2>
            <p className="text-xs" style={{ color: "var(--muted-text)" }}>
              Melindungi kuota API VLM & memori VPS 14 GB selama uji coba live
            </p>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4 mt-4">
          <div>
            <label className="block text-xs font-medium mb-1.5" style={{ color: "var(--body-text)" }}>
              Passcode Penguji Resmi
            </label>
            <input
              type="password"
              autoFocus
              value={passcode}
              onChange={(e) => setPasscode(e.target.value)}
              placeholder="Masukkan kode akses..."
              className="w-full px-3.5 py-2.5 rounded-xl text-sm font-mono-tech outline-none"
              style={{
                backgroundColor: "var(--surface-composer)",
                border: "1px solid var(--hairline-strong)",
                color: "var(--ink)",
              }}
            />
            <p className="text-[11px] mt-1.5" style={{ color: "var(--muted-text)" }}>
              Hint untuk penguji internal: gunakan kode akses tim riset AnyCount. Sesi berlaku selama 30 hari.
            </p>
          </div>

          {error && (
            <div
              className="flex items-center gap-2 px-3 py-2 rounded-lg text-xs"
              style={{
                backgroundColor: "rgba(239, 68, 68, 0.14)",
                color: "var(--status-danger)",
                border: "1px solid rgba(239, 68, 68, 0.3)",
              }}
            >
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          <div className="flex items-center justify-end gap-2 pt-1">
            <button
              type="button"
              onClick={onClose}
              className="px-3.5 py-2 rounded-xl text-xs font-medium cursor-pointer"
              style={{
                backgroundColor: "var(--surface-strong)",
                color: "var(--body-text)",
              }}
            >
              Batal
            </button>
            <button
              type="submit"
              disabled={loading || !passcode.trim()}
              className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-semibold cursor-pointer disabled:opacity-50"
              style={{
                backgroundColor: "var(--nav-active-bg)",
                color: "var(--nav-active-fg)",
              }}
            >
              <ShieldCheck className="w-4 h-4" />
              <span>{loading ? "Memverifikasi..." : "Unlock Studio"}</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
