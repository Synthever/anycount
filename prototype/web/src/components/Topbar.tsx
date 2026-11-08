import React from "react";
import { PanelLeft, Sun, Moon, Lock, Unlock } from "lucide-react";

interface TopbarProps {
  pageTitle: string;
  dark: boolean;
  onToggleTheme: () => void;
  unlocked: boolean;
  onOpenUnlock: () => void;
  onToggleSidebar: () => void;
  onLogout?: () => void;
}

export const Topbar: React.FC<TopbarProps> = ({
  pageTitle,
  dark,
  onToggleTheme,
  unlocked,
  onOpenUnlock,
  onToggleSidebar,
  onLogout,
}) => {
  return (
    <header
      className="h-13 sticky top-0 z-30 px-4 flex items-center justify-between backdrop-blur-md"
      style={{
        backgroundColor: "var(--canvas)",
        borderBottom: "1px solid var(--hairline)",
      }}
    >
      {/* Left: Sidebar toggle + Breadcrumb */}
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={onToggleSidebar}
          aria-label="Toggle sidebar"
          className="p-1.5 rounded-lg cursor-pointer transition-colors hover:opacity-80"
          style={{
            backgroundColor: "var(--surface-card)",
            border: "1px solid var(--hairline)",
            color: "var(--body-text)",
          }}
        >
          <PanelLeft className="w-4 h-4" />
        </button>

        <div className="flex items-center gap-2 text-xs">
          <span style={{ color: "var(--muted-text)" }}>Studio</span>
          <span style={{ color: "var(--muted-text)" }}>/</span>
          <span className="font-medium" style={{ color: "var(--ink)" }}>
            {pageTitle}
          </span>
        </div>
      </div>

      {/* Right: Theme Toggle + Passcode Pill + Avatar */}
      <div className="flex items-center gap-2.5">
        <button
          type="button"
          onClick={onToggleTheme}
          aria-label="Toggle theme"
          className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-full text-xs font-medium cursor-pointer transition-colors"
          style={{
            backgroundColor: "var(--surface-card)",
            border: "1px solid var(--hairline-strong)",
            color: "var(--ink)",
          }}
        >
          {dark ? (
            <>
              <Moon className="w-3.5 h-3.5" style={{ color: "var(--brand-lavender)" }} />
              <span>Theme: Dark</span>
            </>
          ) : (
            <>
              <Sun className="w-3.5 h-3.5" style={{ color: "var(--brand-ochre)" }} />
              <span>Theme: Warm Cream</span>
            </>
          )}
        </button>

        {unlocked ? (
          <button
            type="button"
            onClick={onLogout}
            title="Klik untuk mengunci kembali sesi"
            className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-full text-xs font-mono-tech font-medium cursor-pointer"
            style={{
              backgroundColor: "var(--surface-card)",
              border: "1px solid var(--hairline-strong)",
              color: "var(--brand-emerald)",
            }}
          >
            <Unlock className="w-3.5 h-3.5" />
            <span>Unlocked</span>
          </button>
        ) : (
          <button
            type="button"
            onClick={onOpenUnlock}
            className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-full text-xs font-mono-tech font-medium cursor-pointer transition-opacity hover:opacity-90"
            style={{
              backgroundColor: "var(--surface-card)",
              border: "1px solid var(--hairline-strong)",
              color: "var(--brand-peach)",
            }}
          >
            <Lock className="w-3.5 h-3.5" />
            <span>Locked · Enter Passcode</span>
          </button>
        )}

        <div
          className="w-7 h-7 rounded-full flex items-center justify-center font-mono-tech text-[11px] font-semibold select-none"
          style={{
            backgroundColor: "var(--brand-lavender)",
            color: "#0a0a0a",
          }}
        >
          AC
        </div>
      </div>
    </header>
  );
};
