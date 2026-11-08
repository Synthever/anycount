import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { applyCanvasClick, pointerToImagePoint } from "../components/InteractiveCanvas";
import { MethodResultCard } from "../components/MethodResultCard";
import { Composer } from "../components/Composer";
import { MethodResult } from "../api/client";

describe("InteractiveCanvas dot editing logic", () => {
  it("adds a new point on empty space and removes an existing point when clicked within radius", () => {
    const initial = [
      { x: 100, y: 100 },
      { x: 250, y: 180 },
    ];

    // Click empty area -> +1 point
    const added = applyCanvasClick(initial, 400, 300, 14);
    expect(added.delta).toBe("added");
    expect(added.next.length).toBe(3);

    // Click near (100, 100) -> removes that dot
    const removed = applyCanvasClick(added.next, 105, 98, 14);
    expect(removed.delta).toBe("removed");
    expect(removed.next.length).toBe(2);
    expect(removed.next.some((p) => p.x === 100 && p.y === 100)).toBe(false);
  });

  it("maps pointer coordinates through the rendered image box and clamps to image bounds", () => {
    // An 800x600 image painted into a 400x300 box offset at (40, 20).
    const rect = { left: 40, top: 20, width: 400, height: 300 };

    const center = pointerToImagePoint(rect, 240, 170, 800, 600);
    expect(center.x).toBe(400);
    expect(center.y).toBe(300);
    // 14 screen px of hit radius -> 28 image px at 2x scale.
    expect(center.hitRadiusPx).toBe(28);

    const clamped = pointerToImagePoint(rect, -100, 900, 800, 600);
    expect(clamped.x).toBe(0);
    expect(clamped.y).toBe(600);
  });

  it("renders MethodResultCard m2 cold_start actions and Composer compare presets", () => {
    const onRunValidatorM3 = vi.fn();
    const onAutoTrainPack = vi.fn();

    render(
      <MethodResultCard
        result={{
          methodId: "m2",
          status: "cold_start",
          count: null,
          points: [],
          gridBoxes: null,
          elapsedMs: 4,
          promptTokens: 0,
          completionTokens: 0,
          modelSizeMb: null,
          meta: { message: "Model pack .onnx untuk 'coins' belum dilatih." },
          renderedUrl: "/api/runs/run_1/render/m2",
        }}
        imageUrl="/api/images/sample_coins/raw"
        imageWidth={800}
        imageHeight={600}
        query="coins"
        onRunValidatorM3={onRunValidatorM3}
        onAutoTrainPack={onAutoTrainPack}
      />
    );

    fireEvent.click(screen.getByRole("button", { name: /Run LLM Validator \(M3\)/i }));
    fireEvent.click(screen.getByRole("button", { name: /Auto-Train \.onnx Pack/i }));
    expect(onRunValidatorM3).toHaveBeenCalledTimes(1);
    expect(onAutoTrainPack).toHaveBeenCalledTimes(1);

    const onChangeMethods = vi.fn();
    render(
      <Composer
        samples={[
          {
            id: "sample_apples",
            filename: "apples.jpg",
            defaultQuery: "apples",
            groundTruth: 289,
            width: 1000,
            height: 667,
            url: "/api/images/sample_apples/raw",
          },
        ]}
        selectedImageId="sample_apples"
        selectedImageLabel="apples.jpg"
        query="apples"
        onQueryChange={() => {}}
        compareMode={true}
        onToggleCompareMode={() => {}}
        selectedMethods={["m2", "m3", "m6"]}
        onChangeMethods={onChangeMethods}
        onSelectSample={() => {}}
        onUploadFile={() => {}}
        onSubmitRun={() => {}}
        onToggleHistory={() => {}}
        running={false}
        uploading={false}
      />
    );

    fireEvent.click(screen.getByRole("button", { name: /All 6 Methods/i }));
    expect(onChangeMethods).toHaveBeenCalledWith(["m1", "m2", "m3", "m4", "m5", "m6"]);
  });
});

describe("MethodResultCard live count after SSE updates", () => {
  const baseProps = {
    imageUrl: "/api/images/sample_coins/raw",
    imageWidth: 800,
    imageHeight: 600,
    query: "coins",
  };

  it("shows the completed server count (24) in the Raw drawer instead of the queued 0", () => {
    const queued: MethodResult = {
      methodId: "m3",
      status: "queued",
      count: null,
      points: [],
      gridBoxes: null,
      elapsedMs: null,
      promptTokens: null,
      completionTokens: null,
      modelSizeMb: null,
      meta: {},
      renderedUrl: "/api/runs/run_2/render/m3",
    };

    const completed: MethodResult = {
      ...queued,
      status: "completed",
      count: 24,
      points: Array.from({ length: 24 }, (_, i) => ({ x: 30 + i * 12, y: 120 })),
      elapsedMs: 1240,
      promptTokens: 3243,
      completionTokens: 470,
    };

    const { rerender } = render(<MethodResultCard result={queued} {...baseProps} />);
    // Simulates the SSE transition: same card instance receives new server truth.
    rerender(<MethodResultCard result={completed} {...baseProps} />);

    fireEvent.click(screen.getByRole("button", { name: /Raw/i }));

    const drawer = document.querySelector("pre");
    expect(drawer).toBeTruthy();
    const raw = JSON.parse(drawer?.textContent ?? "{}");
    expect(raw.status).toBe("completed");
    expect(raw.count).toBe(24);
    expect(raw.manualDelta).toBe(0);
  });
});
