import React, { useEffect, useState } from "react";
import { BarChart3, Maximize2, X, Award, Zap, Cpu } from "lucide-react";
import { api, MontageItem } from "../api/client";

interface BenchmarkRow {
  scenario: string;
  query: string;
  gt: string;
  m1: string;
  m2: string;
  m3: string;
  m4: string;
  m5: string;
  m6: string;
  winner: string;
}

const BENCHMARK_TABLE: BenchmarkRow[] = [
  {
    scenario: "cats.jpg",
    query: "cats",
    gt: "2",
    m1: "2 (23.0s)",
    m2: "Cold -> M3",
    m3: "2 (5.71s)",
    m4: "2 (8.4s)",
    m5: "2 (3.20s)",
    m6: "2 (2.9s)",
    winner: "M5 / M3 / M1",
  },
  {
    scenario: "coins.jpg",
    query: "coins",
    gt: "24",
    m1: "24 (18.2s)",
    m2: "Cold -> M3",
    m3: "24 (13.93s)",
    m4: "24 (20.78s)",
    m5: "24 (3.13s)",
    m6: "25 (2.83s)",
    winner: "M5 / M3",
  },
  {
    scenario: "tennis-ball.jpg",
    query: "tennis ball",
    gt: "43",
    m1: "41 (13.6s)",
    m2: "Cold -> M3",
    m3: "42 (7.71s)",
    m4: "42 (10.67s)",
    m5: "40 (3.91s)",
    m6: "40 (3.45s)",
    winner: "M3 (97.7%)",
  },
  {
    scenario: "containers.jpg",
    query: "container",
    gt: "64",
    m1: "64 (28.1s)",
    m2: "Cold -> M4",
    m3: "29 (54.7% err)",
    m4: "65 (31.53s)",
    m5: "69 (2.92s)",
    m6: "61 (2.47s)",
    winner: "M4 / M6 / M1",
  },
  {
    scenario: "Heidelbeeren.jpg",
    query: "blueberry (Color)",
    gt: "~113",
    m1: "125 (13.6s)",
    m2: "Cold -> M3",
    m3: "122 (12.39s)",
    m4: "122 (11.34s)",
    m5: "124 (4.10s)",
    m6: "121 (3.10s)",
    winner: "M6 / M3",
  },
  {
    scenario: "Heidelbeeren-bnw.jpg",
    query: "blueberry (Grayscale)",
    gt: "~113",
    m1: "131 (13.6s)",
    m2: "Cold -> M3",
    m3: "112 (12.10s)",
    m4: "119 (11.48s)",
    m5: "99 (5.12s)",
    m6: "121 (3.88s)",
    winner: "M3 (0.9% err)",
  },
  {
    scenario: "Heidelbeeren-bnw2.jpg",
    query: "blueberry (High-Contrast)",
    gt: "~113",
    m1: "137 (16.6s)",
    m2: "Cold -> M4",
    m3: "128 (13.99s)",
    m4: "127 (10.98s)",
    m5: "105 (5.70s)",
    m6: "128 (4.45s)",
    winner: "M4 / M6",
  },
  {
    scenario: "candy-ball.jpg",
    query: "candy (Color)",
    gt: "~132",
    m1: "128 (13.6s)",
    m2: "Cold -> M3",
    m3: "128 (19.02s)",
    m4: "145 (14.06s)",
    m5: "99 (3.88s)",
    m6: "142 (4.70s)",
    winner: "M3 / M1 (0 double)",
  },
  {
    scenario: "candy-bnw.jpg",
    query: "candy (B&W)",
    gt: "~132",
    m1: "123 (14.1s)",
    m2: "Cold -> M6",
    m3: "116 (12.49s)",
    m4: "144 (12.84s)",
    m5: "132 (5.27s)",
    m6: "133 (3.64s)",
    winner: "M6 (0.8% err)",
  },
  {
    scenario: "apples.jpg",
    query: "apples (Dense Pile)",
    gt: "289",
    m1: "339 (23.0s)",
    m2: "289 (0.54s 2x2)",
    m3: "289 (39.94s)",
    m4: "336 (21.09s)",
    m5: "825 (loop)",
    m6: "336 (5.60s)",
    winner: "M2 (0.54s tiled) & M6 (5.6s)",
  },
];

export const BenchmarkPage: React.FC = () => {
  const [montages, setMontages] = useState<MontageItem[]>([]);
  const [lightbox, setLightbox] = useState<MontageItem | null>(null);

  useEffect(() => {
    api
      .getSamples()
      .then((res) => setMontages(res.montages))
      .catch(() => {});
  }, []);

  return (
    <div className="max-w-7xl mx-auto p-4 md:p-6 space-y-8">
      {/* Header & Executive KPI Cards */}
      <div className="space-y-4">
        <div>
          <div className="inline-flex items-center gap-2 text-xs font-mono-tech uppercase tracking-wider mb-1"
            style={{ color: "var(--brand-pink)" }}
          >
            <BarChart3 className="w-3.5 h-3.5" />
            <span>Spike 000–006 Quantitative & Visual Audit</span>
          </div>
          <h1 className="text-2xl font-semibold display-head">
            Benchmark Gallery — 6 Counting Methods Across 10 Test Images
          </h1>
          <p className="text-xs mt-1 max-w-3xl leading-relaxed" style={{ color: "var(--muted-text)" }}>
            Arsip benchmark riset (Spike 000–006) dari repositori riset AnyCount. Angka pada tabel di bawah adalah
            data riset historis, bukan perilaku prototype saat ini: M2 prototype terukur 289/289 apel
            (MAPE 0% pada satu sampel) dengan 0.15 s untuk foto jarang dan 0.54 s untuk foto padat (mode 2x2 tiled).
            Klik gambar montage untuk memperbesar dan memeriksa penempatan titik per objek.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div
            className="rounded-2xl p-4 space-y-1.5"
            style={{
              backgroundColor: "var(--surface-card)",
              border: "1px solid var(--hairline-strong)",
            }}
          >
            <div className="flex items-center justify-between text-xs font-mono-tech">
              <span style={{ color: "var(--brand-emerald)" }}>M2 · OFFLINE ONNX v2</span>
              <Cpu className="w-4 h-4" style={{ color: "var(--brand-emerald)" }} />
            </div>
            <div className="text-xl font-bold font-mono-tech">0.15s sparse · 0.54s dense · 0% MAPE</div>
            <p className="text-[11px]" style={{ color: "var(--muted-text)" }}>
              Paket model <code>apple_v2_distilled_int8.onnx</code> (2.89 MB) menghitung 289 apel dari GT 289
              (MAPE 0% pada satu sampel uji) dalam 0.54 s mode 2x2 tiled, tanpa internet dan biaya server Rp 0.
            </p>
          </div>

          <div
            className="rounded-2xl p-4 space-y-1.5"
            style={{
              backgroundColor: "var(--surface-card)",
              border: "1px solid var(--hairline-strong)",
            }}
          >
            <div className="flex items-center justify-between text-xs font-mono-tech">
              <span style={{ color: "var(--brand-pink)" }}>M3 & M4 · GEMINI 3.8 FLASH</span>
              <Award className="w-4 h-4" style={{ color: "var(--brand-pink)" }} />
            </div>
            <div className="text-xl font-bold font-mono-tech">Best Per-Instance Grounding</div>
            <p className="text-[11px]" style={{ color: "var(--muted-text)" }}>
              Posisi titik [y, x] paling jujur di pusat tiap benda pada tumpukan &gt;100 objek; menjadi Guru Validator terbaik untuk Auto-Train Metode 2.
            </p>
          </div>

          <div
            className="rounded-2xl p-4 space-y-1.5"
            style={{
              backgroundColor: "var(--surface-card)",
              border: "1px solid var(--hairline-strong)",
            }}
          >
            <div className="flex items-center justify-between text-xs font-mono-tech">
              <span style={{ color: "var(--brand-lavender)" }}>M6 · DEEPSEEK V4.1 2x2 TILED</span>
              <Zap className="w-4 h-4" style={{ color: "var(--brand-lavender)" }} />
            </div>
            <div className="text-xl font-bold font-mono-tech">Avg 3.79s · Fastest Cloud</div>
            <p className="text-[11px]" style={{ color: "var(--muted-text)" }}>
              Pembagian petak 2×2 paralel + Adaptive Point-NMS menghilangkan gejala loop pada 336 apel (5.60s) dan mencapai 0.8% error pada candy-bnw.
            </p>
          </div>
        </div>
      </div>

      {/* Summary Metrics Table */}
      <div
        className="rounded-2xl overflow-hidden"
        style={{
          backgroundColor: "var(--surface-card)",
          border: "1px solid var(--hairline-strong)",
        }}
      >
        <div
          className="px-4 py-3 flex items-center justify-between"
          style={{ borderBottom: "1px solid var(--hairline)" }}
        >
          <span className="text-xs font-semibold">
            Tabel Perbandingan Kuantitatif 6 Metode pada 10 Gambar Uji
            <span
              className="ml-2 font-mono-tech text-[10px] uppercase px-1.5 py-0.5 rounded"
              style={{ backgroundColor: "var(--surface-strong)", color: "var(--muted-text)" }}
              title="Angka 0.21 s / 290 apel / 2.33% MAPE adalah arsip riset pra-prototype; prototype terukur 289/289 pada 0.54 s tiled."
            >
              Data riset historis
            </span>
          </span>
          <span className="font-mono-tech text-[11px]" style={{ color: "var(--muted-text)" }}>
            Format: Count (Latency)
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr
                className="font-mono-tech text-[11px] uppercase"
                style={{
                  backgroundColor: "var(--surface-soft)",
                  color: "var(--muted-text)",
                  borderBottom: "1px solid var(--hairline)",
                }}
              >
                <th className="py-2.5 px-3">Scenario</th>
                <th className="py-2.5 px-3">GT</th>
                <th className="py-2.5 px-3">M1 SAM3</th>
                <th className="py-2.5 px-3">M2 ONNX</th>
                <th className="py-2.5 px-3">M3 Gemini 1x</th>
                <th className="py-2.5 px-3">M4 Gemini 2x2</th>
                <th className="py-2.5 px-3">M5 DeepSeek 1x</th>
                <th className="py-2.5 px-3">M6 DeepSeek 2x2</th>
                <th className="py-2.5 px-3">Best Method</th>
              </tr>
            </thead>
            <tbody className="font-mono-tech text-[11px]">
              {BENCHMARK_TABLE.map((row) => (
                <tr
                  key={row.scenario}
                  style={{ borderBottom: "1px solid var(--hairline)" }}
                >
                  <td className="py-2.5 px-3 font-sans font-medium" style={{ color: "var(--ink)" }}>
                    <div>{row.scenario}</div>
                    <div className="text-[10px]" style={{ color: "var(--muted-text)" }}>
                      &ldquo;{row.query}&rdquo;
                    </div>
                  </td>
                  <td className="py-2.5 px-3 font-bold" style={{ color: "var(--brand-emerald)" }}>
                    {row.gt}
                  </td>
                  <td className="py-2.5 px-3">{row.m1}</td>
                  <td className="py-2.5 px-3">{row.m2}</td>
                  <td className="py-2.5 px-3">{row.m3}</td>
                  <td className="py-2.5 px-3">{row.m4}</td>
                  <td className="py-2.5 px-3">{row.m5}</td>
                  <td className="py-2.5 px-3">{row.m6}</td>
                  <td className="py-2.5 px-3 font-semibold" style={{ color: "var(--brand-lavender)" }}>
                    {row.winner}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Interactive Visual Gallery of 11 Montage JPEGs */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-base font-semibold">
            Visual Side-by-Side Montages ({montages.length} Research Panels)
          </h2>
          <span className="text-xs" style={{ color: "var(--muted-text)" }}>
            Klik kartu untuk membuka Lightbox resolusi penuh
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {montages.map((item) => (
            <div
              key={item.id}
              onClick={() => setLightbox(item)}
              className="rounded-2xl overflow-hidden flex flex-col justify-between cursor-pointer group transition-transform hover:-translate-y-0.5"
              style={{
                backgroundColor: "var(--surface-card)",
                border: "1px solid var(--hairline-strong)",
              }}
            >
              <div className="relative bg-black/40 aspect-video overflow-hidden">
                <img
                  src={item.url}
                  alt={item.title}
                  loading="lazy"
                  className="w-full h-full object-cover group-hover:scale-103 transition-transform"
                />
                <div
                  className="absolute top-2.5 right-2.5 p-1.5 rounded-lg opacity-80 group-hover:opacity-100"
                  style={{
                    backgroundColor: "rgba(10, 10, 12, 0.78)",
                    color: "#ffffff",
                  }}
                >
                  <Maximize2 className="w-3.5 h-3.5" />
                </div>
              </div>

              <div className="p-3.5 space-y-1">
                <div className="text-xs font-semibold" style={{ color: "var(--ink)" }}>
                  {item.title}
                </div>
                <div
                  className="font-mono-tech text-[11px]"
                  style={{ color: "var(--brand-emerald)" }}
                >
                  Best: {item.best}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Lightbox Modal */}
      {lightbox && (
        <div
          onClick={() => setLightbox(null)}
          className="fixed inset-0 z-50 flex items-center justify-center p-4"
          style={{ backgroundColor: "rgba(0, 0, 0, 0.85)", backdropFilter: "blur(6px)" }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="max-w-6xl w-full rounded-2xl overflow-hidden shadow-2xl"
            style={{
              backgroundColor: "var(--surface-card)",
              border: "1px solid var(--hairline-strong)",
            }}
          >
            <div
              className="px-4 py-3 flex items-center justify-between"
              style={{ borderBottom: "1px solid var(--hairline)" }}
            >
              <div>
                <div className="text-sm font-semibold">{lightbox.title}</div>
                <div
                  className="font-mono-tech text-xs"
                  style={{ color: "var(--brand-emerald)" }}
                >
                  Winner: {lightbox.best}
                </div>
              </div>
              <button
                type="button"
                onClick={() => setLightbox(null)}
                aria-label="Close lightbox"
                className="p-1.5 rounded-lg cursor-pointer"
                style={{
                  backgroundColor: "var(--surface-strong)",
                  color: "var(--ink)",
                }}
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="p-4 bg-black flex items-center justify-center max-h-[80vh] overflow-auto">
              <img
                src={lightbox.url}
                alt={lightbox.title}
                className="max-w-full max-h-[75vh] object-contain"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
