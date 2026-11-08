import React from "react";
import { NavLink, useNavigate } from "react-router-dom";
import {
  Sparkles,
  BarChart3,
  Cpu,
  BookOpen,
  CheckCircle2,
  ArrowUpRight,
} from "lucide-react";

interface SidebarProps {
  collapsed: boolean;
  packCount: number;
}

const NAV_GROUPS = [
  {
    group: "STUDIO",
    items: [{ to: "/", label: "Playground", icon: Sparkles }],
  },
  {
    group: "BENCHMARK",
    items: [{ to: "/benchmark", label: "Benchmark Gallery", icon: BarChart3 }],
  },
  {
    group: "ENGINE",
    items: [{ to: "/packs", label: "Model Packs (.onnx)", icon: Cpu }],
  },
  {
    group: "REFERENCE",
    items: [{ to: "/docs", label: "Method Docs & Glossary", icon: BookOpen }],
  },
];

export const Sidebar: React.FC<SidebarProps> = ({ collapsed, packCount }) => {
  const navigate = useNavigate();

  if (collapsed) {
    return null;
  }

  return (
    <aside
      className="w-[248px] shrink-0 h-screen sticky top-0 flex flex-col justify-between select-none"
      style={{
        backgroundColor: "var(--surface-soft)",
        borderRight: "1px solid var(--hairline)",
      }}
    >
      <div className="p-4 space-y-5">
        {/* Brand Block */}
        <div className="flex items-center justify-between px-1.5 pt-1">
          <div className="flex items-center gap-2.5">
            <div
              className="w-8 h-8 rounded-lg flex items-center justify-center font-mono-tech font-semibold text-xs"
              style={{
                backgroundColor: "var(--nav-active-bg)",
                color: "var(--nav-active-fg)",
              }}
            >
              AC
            </div>
            <div>
              <div className="font-semibold text-sm tracking-tight leading-tight">
                AnyCount Studio
              </div>
              <div
                className="font-mono-tech text-[10px] tracking-wider uppercase mt-0.5"
                style={{ color: "var(--muted-text)" }}
              >
                PROTOTYPE / v0.1.0
              </div>
            </div>
          </div>
        </div>

        {/* Active Offline ONNX Pack Card */}
        <div
          className="rounded-xl p-3 transition-colors"
          style={{
            backgroundColor: "var(--surface-card)",
            border: "1px solid var(--hairline-strong)",
          }}
        >
          <div className="flex items-center justify-between mb-1.5">
            <span className="inline-flex items-center gap-1.5 text-[11px] font-medium">
              <CheckCircle2
                className="w-3.5 h-3.5"
                style={{ color: "var(--brand-emerald)" }}
              />
              <span>Offline ONNX Engine</span>
            </span>
            <span
              className="font-mono-tech text-[10px] px-1.5 py-0.5 rounded"
              style={{
                backgroundColor: "var(--surface-strong)",
                color: "var(--brand-emerald)",
              }}
            >
              {packCount} pack{packCount === 1 ? "" : "s"}
            </span>
          </div>
          <div
            className="font-mono-tech text-[11px] truncate"
            style={{ color: "var(--body-text)" }}
          >
            apple · 2.89 MB Ready
          </div>
          <button
            type="button"
            onClick={() => navigate("/packs")}
            className="mt-2 w-full flex items-center justify-between text-[11px] font-medium px-2 py-1 rounded-md cursor-pointer transition-opacity hover:opacity-90"
            style={{
              backgroundColor: "var(--surface-strong)",
              color: "var(--ink)",
            }}
          >
            <span>Manage INT8 Packs</span>
            <ArrowUpRight className="w-3 h-3" />
          </button>
        </div>

        {/* Navigation Groups */}
        <nav className="space-y-4">
          {NAV_GROUPS.map((grp) => (
            <div key={grp.group} className="space-y-1">
              <div
                className="px-2.5 text-[10px] font-mono-tech font-semibold tracking-wider uppercase"
                style={{ color: "var(--muted-text)" }}
              >
                {grp.group}
              </div>
              {grp.items.map((item) => {
                const Icon = item.icon;
                return (
                  <NavLink
                    key={item.to}
                    to={item.to}
                    end={item.to === "/"}
                    className="flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-xs font-medium transition-colors"
                    style={({ isActive }) =>
                      isActive
                        ? {
                            backgroundColor: "var(--nav-active-bg)",
                            color: "var(--nav-active-fg)",
                          }
                        : {
                            color: "var(--body-text)",
                          }
                    }
                  >
                    <Icon className="w-4 h-4 shrink-0" />
                    <span className="truncate">{item.label}</span>
                  </NavLink>
                );
              })}
            </div>
          ))}
        </nav>
      </div>

      {/* Footer Info */}
      <div
        className="p-4 text-[11px] space-y-1"
        style={{
          borderTop: "1px solid var(--hairline)",
          color: "var(--muted-text)",
        }}
      >
        <div className="flex items-center justify-between font-mono-tech">
          <span>6 Counting Methods</span>
          <span>M1–M6</span>
        </div>
        <div className="text-[10px] leading-relaxed">
          SAM3 · ONNX INT8 · Gemini 3.8 · DeepSeek V4.1
        </div>
      </div>
    </aside>
  );
};
