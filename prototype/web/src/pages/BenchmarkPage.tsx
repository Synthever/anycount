import React, { useEffect, useState } from "react";
import { Maximize2, X } from "lucide-react";
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
  m7: string;
  winner: string;
}

const BENCHMARK_TABLE: BenchmarkRow[] = [
  {
    scenario: "cats.jpg",
    query: "cats",
    gt: "2",
    m1: "2 (23.0s)",
    m2: "Cold start",
    m3: "Cold start",
    m4: "2 (5.71s)",
    m5: "2 (8.4s)",
    m6: "2 (3.20s)",
    m7: "2 (2.9s)",
    winner: "M6 / M4 / M1",
  },
  {
    scenario: "coins.jpg",
    query: "coins",
    gt: "24",
    m1: "24 (18.2s)",
    m2: "Cold start",
    m3: "Cold start",
    m4: "24 (13.93s)",
    m5: "24 (20.78s)",
    m6: "24 (3.13s)",
    m7: "25 (2.83s)",
    winner: "M6 / M4",
  },
  {
    scenario: "tennis-ball.jpg",
    query: "tennis ball",
    gt: "43",
    m1: "41 (13.6s)",
    m2: "Cold start",
    m3: "Cold start",
    m4: "42 (7.71s)",
    m5: "42 (10.67s)",
    m6: "40 (3.91s)",
    m7: "40 (3.45s)",
    winner: "M4 (97.7%)",
  },
  {
    scenario: "containers.jpg",
    query: "container",
    gt: "64",
    m1: "64 (28.1s)",
    m2: "Cold start",
    m3: "Cold start",
    m4: "29 (54.7% err)",
    m5: "65 (31.53s)",
    m6: "69 (2.92s)",
    m7: "61 (2.47s)",
    winner: "M5 / M7 / M1",
  },
  {
    scenario: "Heidelbeeren.jpg",
    query: "blueberry (Color)",
    gt: "~113",
    m1: "125 (13.6s)",
    m2: "Cold start",
    m3: "Cold start",
    m4: "122 (12.39s)",
    m5: "122 (11.34s)",
    m6: "124 (4.10s)",
    m7: "121 (3.10s)",
    winner: "M7 / M4",
  },
  {
    scenario: "Heidelbeeren-bnw.jpg",
    query: "blueberry (Grayscale)",
    gt: "~113",
    m1: "131 (13.6s)",
    m2: "Cold start",
    m3: "Cold start",
    m4: "112 (12.10s)",
    m5: "119 (11.48s)",
    m6: "99 (5.12s)",
    m7: "121 (3.88s)",
    winner: "M4 (0.9% err)",
  },
  {
    scenario: "Heidelbeeren-bnw2.jpg",
    query: "blueberry (High-Contrast)",
    gt: "~113",
    m1: "137 (16.6s)",
    m2: "Cold start",
    m3: "Cold start",
    m4: "128 (13.99s)",
    m5: "127 (10.98s)",
    m6: "105 (5.70s)",
    m7: "128 (4.45s)",
    winner: "M5 / M7",
  },
  {
    scenario: "candy-ball.jpg",
    query: "candy (Color)",
    gt: "~132",
    m1: "128 (13.6s)",
    m2: "Cold start",
    m3: "Cold start",
    m4: "128 (19.02s)",
    m5: "145 (14.06s)",
    m6: "99 (3.88s)",
    m7: "142 (4.70s)",
    winner: "M4 / M1 (0 double)",
  },
  {
    scenario: "candy-bnw.jpg",
    query: "candy (B&W)",
    gt: "~132",
    m1: "123 (14.1s)",
    m2: "Cold start",
    m3: "Cold start",
    m4: "116 (12.49s)",
    m5: "144 (12.84s)",
    m6: "132 (5.27s)",
    m7: "133 (3.64s)",
    winner: "M7 (0.8% err)",
  },
  {
    scenario: "apples.jpg",
    query: "apples (Dense Pile)",
    gt: "289",
    m1: "339 (23.0s)",
    m2: "324 (0.15s single)",
    m3: "289 (0.54s 2x2)",
    m4: "289 (39.94s)",
    m5: "336 (21.09s)",
    m6: "825 (loop)",
    m7: "336 (5.60s)",
    winner: "M3 (0%) & M7 (5.6s)",
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
      {/* Page header (the spike audit KPI strip was retired) */}
      <div>
        <h1 className="text-2xl font-semibold display-head">
          Benchmark Gallery — 7 Counting Methods Across 10 Test Images
        </h1>
        <p className="text-xs mt-1 max-w-3xl leading-relaxed" style={{ color: "var(--muted-text)" }}>
          Arsip benchmark riset (Spike 000–006) dari repositori riset AnyCount. Angka pada tabel di bawah adalah
          data riset historis, bukan perilaku prototype saat ini: M3 prototype terukur 289/289 apel
          (MAPE 0% pada satu sampel) dalam 0.54 s mode 2x2 tiled, sedangkan M2 single-pass terukur 324 titik
          (+12.1% dari GT) dalam ~0.15 s. Klik gambar montage untuk memperbesar dan memeriksa penempatan titik per objek.
        </p>
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
            Tabel Perbandingan Kuantitatif 7 Metode pada 10 Gambar Uji
            <span
              className="ml-2 font-mono-tech text-[10px] uppercase px-1.5 py-0.5 rounded"
              style={{ backgroundColor: "var(--surface-strong)", color: "var(--muted-text)" }}
              title="Angka 0.21 s / 290 apel / 2.33% MAPE adalah arsip riset pra-prototype; prototype terukur 324 titik (M2 single) dan 289/289 pada 0.54 s tiled (M3)."
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
                <th className="py-2.5 px-3">M2 ONNX 1x</th>
                <th className="py-2.5 px-3">M3 ONNX 2x2</th>
                <th className="py-2.5 px-3">M4 Gemini 1x</th>
                <th className="py-2.5 px-3">M5 Gemini 2x2</th>
                <th className="py-2.5 px-3">M6 DeepSeek 1x</th>
                <th className="py-2.5 px-3">M7 DeepSeek 2x2</th>
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
                  <td className="py-2.5 px-3">{row.m7}</td>
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
