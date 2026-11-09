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
    inverted: false,
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
  return {
    run,
    createRun: vi.fn(
      async (_imageId: string, _query: string, _methods: string[], _invert: boolean) => run
    ),
  };
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
    expect(screen.getByText(/Siap Menguji 7 Metode/)).toBeTruthy();
  });

  it("shows Upload gambar by default and the picked preset name afterwards", async () => {
    const props = { onRequireUnlock: () => {}, onPacksChanged: () => {} };
    render(<PlaygroundPage unlocked={true} {...props} />);

    expect(await screen.findByRole("button", { name: /Upload gambar/i })).toBeTruthy();

    fireEvent.click(await screen.findByRole("button", { name: /^apples$/i }));

    await waitFor(() =>
      expect(screen.getAllByRole("button", { name: /^apples$/i }).length).toBe(2)
    );
  });

  it("sends invert=true when the colour-inversion toggle is on", async () => {
    hoisted.createRun.mockClear();
    render(
      <PlaygroundPage
        unlocked={true}
        onRequireUnlock={() => {}}
        onPacksChanged={() => {}}
      />
    );
    await screen.findByRole("button", { name: /upload gambar/i });

    fireEvent.click(screen.getByRole("button", { name: /invert warna/i }));
    expect(
      screen.getByRole("button", { name: /invert warna/i }).getAttribute("aria-pressed")
    ).toBe("true");

    fireEvent.click(screen.getByRole("button", { name: /run counting methods/i }));

    await waitFor(() => expect(hoisted.createRun).toHaveBeenCalled());
    expect(hoisted.createRun.mock.calls.at(-1)?.[3]).toBe(true);
  });

  it("marks a run that counted on an inverted image", async () => {
    hoisted.createRun.mockClear();
    hoisted.createRun.mockResolvedValueOnce({ ...hoisted.run, inverted: true });
    render(
      <PlaygroundPage
        unlocked={true}
        onRequireUnlock={() => {}}
        onPacksChanged={() => {}}
      />
    );
    await screen.findByRole("button", { name: /upload gambar/i });

    fireEvent.click(screen.getByRole("button", { name: /run counting methods/i }));

    expect(await screen.findByText("Inverted")).toBeTruthy();
  });
});
