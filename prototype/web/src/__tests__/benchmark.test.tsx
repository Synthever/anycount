import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";

vi.mock("../api/client", () => ({
  api: {
    getSamples: async () => ({ samples: [], montages: [] }),
  },
}));

import { BenchmarkPage } from "../pages/BenchmarkPage";

describe("BenchmarkPage revision", () => {
  it("drops the spike audit header/KPI strip and shows the M1..M7 table", async () => {
    render(<BenchmarkPage />);

    expect(await screen.findByText(/7 Counting Methods Across 10 Test Images/)).toBeTruthy();
    expect(screen.queryByText("Spike 000–006 Quantitative & Visual Audit")).toBeNull();
    expect(screen.queryByText(/Quantitative & Visual Audit/i)).toBeNull();
    expect(screen.queryByText(/0.15s sparse · 0.54s dense · 0% MAPE/)).toBeNull();
    expect(screen.queryByText(/M6 · DEEPSEEK V4.1 2x2 TILED/)).toBeNull();

    for (const label of [
      "M1 SAM3",
      "M2 ONNX 1x",
      "M3 ONNX 2x2",
      "M4 Gemini 1x",
      "M5 Gemini 2x2",
      "M6 DeepSeek 1x",
      "M7 DeepSeek 2x2",
    ]) {
      expect(screen.getByText(label)).toBeTruthy();
    }

    // Measured prototype numbers stay visible for the apples row.
    expect(screen.getByText("324 (0.15s single)")).toBeTruthy();
    expect(screen.getByText("289 (0.54s 2x2)")).toBeTruthy();
  });
});
