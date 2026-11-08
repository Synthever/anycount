import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { PlaygroundPage } from "../pages/PlaygroundPage";

// A run as the server returns it once a method finished.
const hoisted = vi.hoisted(() => {
  const run = {
    runId: "run_lock_test",
    imageId: "sample_apples",
    imageUrl: "/api/images/sample_apples/raw",
    imageWidth: 1000,
    imageHeight: 667,
    query: "apples",
    status: "completed",
    createdAt: "2026-11-08T10:00:00Z",
    methods: [
      {
        methodId: "m2" as const,
        status: "completed" as const,
        count: 289,
        points: [
          { x: 120, y: 140 },
          { x: 420, y: 310 },
        ],
        gridBoxes: null,
        elapsedMs: 540,
        promptTokens: 0,
        completionTokens: 0,
        modelSizeMb: 2.89,
        meta: { packSlug: "apple", mode: "tiled_2x2_640+point_nms" },
        renderedUrl: "/api/runs/run_lock_test/render/m2",
      },
    ],
  };
  return { run, createRun: vi.fn(async () => run) };
});

vi.mock("../api/client", () => ({
  api: {
    getSamples: async () => ({
      samples: [
        {
          id: "sample_apples",
          filename: "apples.jpg",
          defaultQuery: "apples",
          groundTruth: 289,
          width: 1000,
          height: 667,
          url: "/api/images/sample_apples/raw",
        },
      ],
    }),
    listRuns: async () => ({ runs: [] }),
    createRun: hoisted.createRun,
    subscribeRun: () => () => {},
    uploadImage: vi.fn(),
    trainPack: vi.fn(),
  },
}));

describe("Playground session gating", () => {
  it("clears a fetched run (points + render) when the session locks", async () => {
    const props = { onRequireUnlock: () => {}, onPacksChanged: () => {} };
    const { rerender } = render(<PlaygroundPage unlocked={true} {...props} />);

    fireEvent.click(
      await screen.findByRole("button", { name: /Run Selected Methods/i })
    );
    await waitFor(() => expect(screen.getByText(/289 objects/)).toBeTruthy());

    rerender(<PlaygroundPage unlocked={false} {...props} />);

    await waitFor(() => expect(screen.queryByText(/289 objects/)).toBeNull());
    expect(screen.getByText(/Siap Menguji 6 Metode/)).toBeTruthy();
  });
});
