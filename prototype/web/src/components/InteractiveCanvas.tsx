import React, { useRef, useState, useEffect } from "react";
import { Undo2, RotateCcw, Download, MousePointerClick } from "lucide-react";
import { Point2D } from "../api/client";

export function applyCanvasClick(
  points: Point2D[],
  clickX: number,
  clickY: number,
  hitRadiusPx: number
): { next: Point2D[]; delta: "added" | "removed" } {
  let bestIdx = -1;
  let bestDist = Infinity;
  points.forEach((pt, idx) => {
    const d = Math.hypot(pt.x - clickX, pt.y - clickY);
    if (d <= hitRadiusPx && d < bestDist) {
      bestDist = d;
      bestIdx = idx;
    }
  });
  if (bestIdx >= 0) {
    return {
      next: points.filter((_, idx) => idx !== bestIdx),
      delta: "removed",
    };
  }
  return {
    next: [...points, { x: Math.round(clickX * 100) / 100, y: Math.round(clickY * 100) / 100 }],
    delta: "added",
  };
}

export interface RectLike {
  left: number;
  top: number;
  width: number;
  height: number;
}

/**
 * Map a pointer event (client coordinates) onto the source-image coordinate
 * space. The rect passed in must be the rendered <img> box, which is exactly
 * the painted image because the image is not letterboxed by `object-contain`.
 * Result is clamped to [0, imageWidth] x [0, imageHeight].
 */
export function pointerToImagePoint(
  rect: RectLike,
  clientX: number,
  clientY: number,
  imageWidth: number,
  imageHeight: number
): { x: number; y: number; hitRadiusPx: number } {
  const w = imageWidth > 0 ? imageWidth : 1;
  const h = imageHeight > 0 ? imageHeight : 1;
  const safeW = rect.width > 0 ? rect.width : w;
  const safeH = rect.height > 0 ? rect.height : h;
  const scaleX = w / safeW;
  const scaleY = h / safeH;
  return {
    x: Math.min(w, Math.max(0, (clientX - rect.left) * scaleX)),
    y: Math.min(h, Math.max(0, (clientY - rect.top) * scaleY)),
    // Screen px -> image px for the 14px hit radius, using the looser axis.
    hitRadiusPx: 14 * Math.max(scaleX, scaleY),
  };
}

interface InteractiveCanvasProps {
  imageUrl: string;
  imageWidth: number;
  imageHeight: number;
  query: string;
  initialPoints: Point2D[];
  gridBoxes: [number, number, number, number][] | null;
  editMode: boolean;
  showGrid: boolean;
  methodId: string;
  onPointsChange?: (currentPoints: Point2D[], manualDelta: number) => void;
}

export const InteractiveCanvas: React.FC<InteractiveCanvasProps> = ({
  imageUrl,
  imageWidth,
  imageHeight,
  query,
  initialPoints,
  gridBoxes,
  editMode,
  showGrid,
  methodId,
  onPointsChange,
}) => {
  const [history, setHistory] = useState<Point2D[][]>([initialPoints]);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);

  useEffect(() => {
    setHistory([initialPoints]);
  }, [initialPoints]);

  const currentPoints = history[history.length - 1] ?? initialPoints;
  const effectiveCount = currentPoints.length;
  const manualDelta = effectiveCount - initialPoints.length;

  const updatePoints = (next: Point2D[]) => {
    const nextHist = [...history, next];
    setHistory(nextHist);
    onPointsChange?.(next, next.length - initialPoints.length);
  };

  const handleUndo = () => {
    if (history.length <= 1) return;
    const nextHist = history.slice(0, -1);
    setHistory(nextHist);
    const latest = nextHist[nextHist.length - 1];
    onPointsChange?.(latest, latest.length - initialPoints.length);
  };

  const handleReset = () => {
    setHistory([initialPoints]);
    onPointsChange?.(initialPoints, 0);
  };

  const handleSvgClick = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!editMode) return;
    // Measure the rendered <img> box: the overlay is sized to the image, so the
    // image rect (not the outer frame) is the authoritative click surface.
    const imgRect = imgRef.current?.getBoundingClientRect();
    const fallbackRect = svgRef.current?.getBoundingClientRect();
    const rect =
      imgRect && imgRect.width > 0 && imgRect.height > 0 ? imgRect : fallbackRect;
    if (!rect || rect.width <= 0 || rect.height <= 0) return;

    const w = imageWidth || imgRef.current?.naturalWidth || 800;
    const h = imageHeight || imgRef.current?.naturalHeight || 600;
    const { x: clickX, y: clickY, hitRadiusPx } = pointerToImagePoint(
      rect,
      e.clientX,
      e.clientY,
      w,
      h
    );

    const { next } = applyCanvasClick(currentPoints, clickX, clickY, hitRadiusPx);
    updatePoints(next);
  };

  const handleExportEditedJpg = () => {
    const imgEl = imgRef.current;
    if (!imgEl) return;

    const canvas = document.createElement("canvas");
    const w = imageWidth || imgEl.naturalWidth || 800;
    const h = imageHeight || imgEl.naturalHeight || 600;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(imgEl, 0, 0, w, h);

    // Optional 2x2 grid boxes if visible
    if (showGrid && gridBoxes && gridBoxes.length > 0) {
      ctx.save();
      ctx.strokeStyle = "rgba(184, 164, 237, 0.85)";
      ctx.lineWidth = Math.max(2, Math.round(Math.min(w, h) / 300));
      ctx.setLineDash([8, 6]);
      for (const [x1, y1, x2, y2] of gridBoxes) {
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      }
      ctx.restore();
    }

    // Count-Anything style red (#e62823) dots with white (#ffffff) 2px ring
    const radius = Math.max(5, Math.round(Math.min(w, h) / 140));
    for (const pt of currentPoints) {
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, radius + 2, 0, Math.PI * 2);
      ctx.fillStyle = "#ffffff";
      ctx.fill();

      ctx.beginPath();
      ctx.arc(pt.x, pt.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = "#e62823";
      ctx.fill();
    }

    // Top-left dark translucent counter badge
    const label = `${query}: ${effectiveCount}`;
    const fontSize = Math.max(14, Math.round(Math.min(w, h) / 36));
    ctx.font = `600 ${fontSize}px Inter, sans-serif`;
    const textMetrics = ctx.measureText(label);
    const padX = Math.round(fontSize * 0.7);
    const padY = Math.round(fontSize * 0.45);
    const boxW = textMetrics.width + padX * 2;
    const boxH = fontSize + padY * 2;

    ctx.fillStyle = "rgba(10, 10, 12, 0.78)";
    ctx.fillRect(12, 12, boxW, boxH);
    ctx.fillStyle = "#ffffff";
    ctx.textBaseline = "middle";
    ctx.fillText(label, 12 + padX, 12 + boxH / 2);

    const dataUrl = canvas.toDataURL("image/jpeg", 0.92);
    const a = document.createElement("a");
    a.href = dataUrl;
    a.download = `anycount_${methodId}_${query.replace(/\s+/g, "_")}_${effectiveCount}.jpg`;
    a.click();
  };

  const dotRadius = Math.max(5, Math.round(Math.min(imageWidth || 800, imageHeight || 600) / 135));
  const strokeW = Math.max(2, Math.round(dotRadius * 0.38));

  return (
    <div className="space-y-2">
      {/* Image + SVG overlay. The inner box shrink-wraps the rendered <img>, so the
          overlay is exactly the size of the picture (no object-contain letterbox). */}
      <div className="w-full flex justify-center">
        <div
          className="relative inline-block max-w-full overflow-hidden rounded-xl select-none"
          style={{
            backgroundColor: "#09090b",
            border: editMode
              ? "2px solid var(--brand-pink)"
              : "1px solid var(--hairline)",
          }}
        >
          <img
            ref={imgRef}
            src={imageUrl}
            alt={query}
            crossOrigin="anonymous"
            className="block w-auto h-auto max-w-full max-h-[520px]"
          />

          {/* SVG Overlay aligned with image coordinate space */}
          <svg
            ref={svgRef}
            viewBox={`0 0 ${imageWidth || 800} ${imageHeight || 600}`}
            onClick={handleSvgClick}
            className={`absolute inset-0 w-full h-full ${
              editMode ? "cursor-crosshair" : "cursor-default"
            }`}
            preserveAspectRatio="xMidYMid meet"
          >
            {/* 2x2 Grid Overlay */}
            {showGrid &&
              gridBoxes &&
              gridBoxes.map(([x1, y1, x2, y2], idx) => (
                <g key={`grid-${idx}`}>
                  <rect
                    x={x1}
                    y={y1}
                    width={Math.max(0, x2 - x1)}
                    height={Math.max(0, y2 - y1)}
                    fill="none"
                    stroke="#b8a4ed"
                    strokeWidth={Math.max(2, Math.round(dotRadius * 0.45))}
                    strokeDasharray="10 6"
                    opacity={0.9}
                  />
                </g>
              ))}

            {/* Count-Anything Red-White Dots */}
            {currentPoints.map((pt, idx) => (
              <circle
                key={`pt-${idx}-${pt.x}-${pt.y}`}
                cx={pt.x}
                cy={pt.y}
                r={dotRadius}
                fill="#e62823"
                stroke="#ffffff"
                strokeWidth={strokeW}
              />
            ))}
          </svg>

          {/* Top-Left Dark Translucent Counter Badge */}
          <div
            className="absolute top-3 left-3 px-2.5 py-1 rounded-lg font-mono-tech text-xs font-semibold flex items-center gap-2 pointer-events-none"
            style={{
              backgroundColor: "rgba(10, 10, 12, 0.82)",
              color: "#ffffff",
              border: "1px solid rgba(255, 255, 255, 0.16)",
            }}
          >
            <span className="inline-block w-2 h-2 rounded-full bg-[#e62823] ring-1 ring-white" />
            <span>
              {query}: {effectiveCount}
            </span>
            {manualDelta !== 0 && (
              <span
                className="text-[11px] px-1.5 py-0.5 rounded"
                style={{
                  backgroundColor: "rgba(255, 77, 139, 0.25)",
                  color: "#ff85b3",
                }}
              >
                {manualDelta > 0 ? `+${manualDelta}` : manualDelta} manual
              </span>
            )}
          </div>

          {/* Edit Mode Floating Pill */}
          {editMode && (
            <div
              className="absolute bottom-3 left-3 px-2.5 py-1 rounded-lg text-[11px] font-medium flex items-center gap-1.5 pointer-events-none"
              style={{
                backgroundColor: "rgba(10, 10, 12, 0.85)",
                color: "#fffaf0",
                border: "1px solid var(--brand-pink)",
              }}
            >
              <MousePointerClick className="w-3.5 h-3.5 text-[#ff4d8b]" />
              <span>Klik ruang kosong: +1 titik · Klik titik merah (radius 14px): -1 hapus</span>
            </div>
          )}
        </div>
      </div>

      {/* Edit Mode Toolbar (Undo / Reset / Export Edited JPG) */}
      {(editMode || history.length > 1) && (
        <div
          className="flex flex-wrap items-center justify-between gap-2 px-3 py-2 rounded-xl text-xs"
          style={{
            backgroundColor: "var(--surface-soft)",
            border: "1px solid var(--hairline)",
          }}
        >
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={handleUndo}
              disabled={history.length <= 1}
              className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg font-medium cursor-pointer disabled:opacity-40"
              style={{
                backgroundColor: "var(--surface-strong)",
                color: "var(--ink)",
              }}
            >
              <Undo2 className="w-3.5 h-3.5" />
              <span>Undo</span>
            </button>
            <button
              type="button"
              onClick={handleReset}
              disabled={history.length <= 1}
              className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg font-medium cursor-pointer disabled:opacity-40"
              style={{
                backgroundColor: "var(--surface-strong)",
                color: "var(--body-text)",
              }}
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Reset ({initialPoints.length})</span>
            </button>
          </div>

          <button
            type="button"
            onClick={handleExportEditedJpg}
            className="inline-flex items-center gap-1.5 px-3 py-1 rounded-lg font-medium cursor-pointer"
            style={{
              backgroundColor: "var(--brand-pink)",
              color: "#ffffff",
            }}
          >
            <Download className="w-3.5 h-3.5" />
            <span>Export Edited JPG</span>
          </button>
        </div>
      )}
    </div>
  );
};
