import React, { useEffect, useState } from "react";
import { BookOpen, Search, Clock, HardDrive, CheckCircle2 } from "lucide-react";
import { api, MethodDocItem, GlossaryItem } from "../api/client";

const BRAND_TOKEN_MAP: Record<string, { bg: string; text: string }> = {
  pink: { bg: "var(--brand-pink)", text: "#0a0a0a" },
  teal: { bg: "var(--brand-teal)", text: "#fffaf0" },
  lavender: { bg: "var(--brand-lavender)", text: "#0a0a0a" },
  peach: { bg: "var(--brand-peach)", text: "#0a0a0a" },
  ochre: { bg: "var(--brand-ochre)", text: "#0a0a0a" },
  mint: { bg: "var(--brand-mint)", text: "#0a0a0a" },
  emerald: { bg: "var(--brand-emerald)", text: "#fffaf0" },
};

export const DocsGlossaryPage: React.FC = () => {
  const [methods, setMethods] = useState<MethodDocItem[]>([]);
  const [glossary, setGlossary] = useState<GlossaryItem[]>([]);
  const [search, setSearch] = useState<string>("");

  useEffect(() => {
    api
      .getDocs()
      .then((res) => {
        setMethods(res.methods);
        setGlossary(res.glossary);
      })
      .catch(() => {});
  }, []);

  const filteredGlossary = glossary.filter(
    (g) =>
      g.term.toLowerCase().includes(search.toLowerCase()) ||
      g.plain.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="max-w-6xl mx-auto p-4 md:p-6 space-y-8">
      {/* Header */}
      <div>
        <div
          className="inline-flex items-center gap-2 text-xs font-mono-tech uppercase tracking-wider mb-1"
          style={{ color: "var(--brand-lavender)" }}
        >
          <BookOpen className="w-3.5 h-3.5" />
          <span>Architecture Reference & Plain-Indonesian Guide</span>
        </div>
        <h1 className="text-2xl font-semibold display-head">
          Method Docs (M1–M7) & Kamus Istilah Awam
        </h1>
        <p
          className="text-xs mt-1 max-w-2xl leading-relaxed"
          style={{ color: "var(--muted-text)" }}
        >
          Panduan lengkap cara kerja ketujuh metode penghitung objek AnyCount beserta kamus 12 istilah teknis dalam bahasa sehari-hari.
        </p>
      </div>

      {/* 6 Method Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {methods.map((m) => {
          const palette = BRAND_TOKEN_MAP[m.color] ?? BRAND_TOKEN_MAP.lavender;
          return (
            <div
              key={m.id}
              className="rounded-2xl p-5 flex flex-col justify-between space-y-4"
              style={{
                backgroundColor: "var(--surface-card)",
                border: "1px solid var(--hairline-strong)",
              }}
            >
              <div className="space-y-3">
                <div className="flex items-center justify-between gap-2">
                  <span
                    className="font-mono-tech text-xs font-bold px-2.5 py-1 rounded-lg"
                    style={{
                      backgroundColor: palette.bg,
                      color: palette.text,
                    }}
                  >
                    {m.code}
                  </span>
                  <span
                    className="font-mono-tech text-[11px] px-2 py-0.5 rounded-full"
                    style={{
                      backgroundColor: "var(--surface-strong)",
                      color: "var(--ink)",
                    }}
                  >
                    {m.badge}
                  </span>
                </div>

                <h2 className="text-sm font-semibold">{m.name}</h2>
                <p
                  className="text-xs leading-relaxed"
                  style={{ color: "var(--body-text)" }}
                >
                  {m.summary}
                </p>
              </div>

              <div
                className="pt-3 space-y-2 text-[11px]"
                style={{ borderTop: "1px solid var(--hairline)" }}
              >
                <div className="flex items-center justify-between font-mono-tech">
                  <span className="inline-flex items-center gap-1" style={{ color: "var(--muted-text)" }}>
                    <Clock className="w-3 h-3" />
                    Speed:
                  </span>
                  <span>{m.speed}</span>
                </div>
                <div className="flex items-center justify-between font-mono-tech">
                  <span className="inline-flex items-center gap-1" style={{ color: "var(--muted-text)" }}>
                    <HardDrive className="w-3 h-3" />
                    Footprint:
                  </span>
                  <span>{m.footprint}</span>
                </div>
                <div
                  className="p-2 rounded-lg text-[11px] flex items-start gap-1.5 mt-2"
                  style={{
                    backgroundColor: "var(--surface-soft)",
                    color: "var(--ink)",
                  }}
                >
                  <CheckCircle2
                    className="w-3.5 h-3.5 shrink-0 mt-0.5"
                    style={{ color: "var(--brand-emerald)" }}
                  />
                  <span>
                    <strong>Cocok untuk:</strong> {m.bestFor}
                  </span>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Searchable 12-item Glossary */}
      <div className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-base font-semibold">
              Kamus Istilah Teknis ({filteredGlossary.length} / {glossary.length} istilah)
            </h2>
            <p className="text-xs" style={{ color: "var(--muted-text)" }}>
              Penjelasan sederhana untuk pembaca non-teknis maupun penguji produk
            </p>
          </div>

          <div
            className="flex items-center gap-2 px-3 py-1.5 rounded-xl w-full sm:w-72"
            style={{
              backgroundColor: "var(--surface-card)",
              border: "1px solid var(--hairline-strong)",
            }}
          >
            <Search className="w-3.5 h-3.5" style={{ color: "var(--muted-text)" }} />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Cari istilah (mis. NMS, ONNX, Tiled)..."
              className="w-full text-xs bg-transparent outline-none"
              style={{ color: "var(--ink)" }}
            />
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {filteredGlossary.map((item) => (
            <div
              key={item.term}
              className="rounded-xl p-4 space-y-1"
              style={{
                backgroundColor: "var(--surface-card)",
                border: "1px solid var(--hairline)",
              }}
            >
              <div
                className="font-mono-tech text-xs font-semibold"
                style={{ color: "var(--brand-lavender)" }}
              >
                {item.term}
              </div>
              <p
                className="text-xs leading-relaxed"
                style={{ color: "var(--body-text)" }}
              >
                {item.plain}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
