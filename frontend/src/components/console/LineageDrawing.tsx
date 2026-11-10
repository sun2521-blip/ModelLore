import type { Lineage } from "../../lib/api";
import { shortHash } from "../../lib/format";

// Slab geometry, in viewBox units: a rhombus top face with a thickness.
const CX = 320;
const HALF_W = 150;
const HALF_D = 75;
const THICKNESS = 22;

type Point = [number, number];
type LayerName = "model" | "run" | "config" | "dataset";

const LAYERS: Array<{ name: LayerName; title: string; side: "left" | "right" }> = [
  { name: "model", title: "MODEL ARTIFACT", side: "right" },
  { name: "run", title: "TRAINING RUN", side: "left" },
  { name: "config", title: "CONFIGURATION", side: "right" },
  { name: "dataset", title: "DATASET", side: "left" },
];

const EXPLODED_TOP = 106;
const EXPLODED_GAP = 126;
const STACKED_TOP = 288;
const STACKED_GAP = 30;

/** A point on a slab's top face: u runs north to east, v runs north to west. */
function onFace(cy: number, u: number, v: number): Point {
  const north: Point = [CX, cy - HALF_D];
  const east: Point = [CX + HALF_W, cy];
  const west: Point = [CX - HALF_W, cy];
  return [
    north[0] + u * (east[0] - north[0]) + v * (west[0] - north[0]),
    north[1] + u * (east[1] - north[1]) + v * (west[1] - north[1]),
  ];
}

const points = (list: Point[]) => list.map(([x, y]) => `${x},${y}`).join(" ");

function Slab({ cy }: { cy: number }) {
  const north: Point = [CX, cy - HALF_D];
  const east: Point = [CX + HALF_W, cy];
  const south: Point = [CX, cy + HALF_D];
  const west: Point = [CX - HALF_W, cy];
  const down = (point: Point): Point => [point[0], point[1] + THICKNESS];

  return (
    <g stroke="var(--ink)" strokeWidth={1.5} strokeLinejoin="round">
      <polygon points={points([west, south, down(south), down(west)])} fill="var(--olive)" />
      <polygon points={points([south, east, down(east), down(south)])} fill="var(--olive-deep)" />
      <polygon points={points([north, east, south, west])} fill="var(--sage-light)" />
    </g>
  );
}

function Motif({ name, cy, verified }: { name: LayerName; cy: number; verified: boolean }) {
  if (name === "dataset") {
    const rows = [0.18, 0.34, 0.5, 0.66, 0.82];
    const columns = [0.2, 0.4, 0.6, 0.8];
    return (
      <g stroke="var(--ink)" strokeWidth={1.2} fill="none">
        {rows.map((v) => {
          const [a, b] = [onFace(cy, 0.12, v), onFace(cy, 0.88, v)];
          return <line key={`r${v}`} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} />;
        })}
        {columns.map((u) => {
          const [a, b] = [onFace(cy, u, 0.12), onFace(cy, u, 0.88)];
          return <line key={`c${u}`} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} />;
        })}
      </g>
    );
  }

  if (name === "config") {
    const tracks = [
      { v: 0.3, knob: 0.62 },
      { v: 0.5, knob: 0.34 },
      { v: 0.7, knob: 0.74 },
    ];
    return (
      <g stroke="var(--ink)">
        {tracks.map(({ v, knob }) => {
          const [a, b] = [onFace(cy, 0.14, v), onFace(cy, 0.86, v)];
          const [kx, ky] = onFace(cy, knob, v);
          return (
            <g key={v}>
              <line x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} strokeWidth={3} strokeLinecap="round" />
              <ellipse cx={kx} cy={ky} rx={9} ry={5} fill="var(--sage)" strokeWidth={1.5} />
            </g>
          );
        })}
      </g>
    );
  }

  if (name === "run") {
    const levels = [0.35, 0.6, 0.45, 0.85, 0.55, 0.75, 0.3, 0.65, 0.95, 0.5, 0.7, 0.4];
    return (
      <g stroke="var(--ink)" strokeWidth={4} strokeLinecap="round">
        {levels.map((level, index) => {
          const [x, y] = onFace(cy, 0.12 + (index / (levels.length - 1)) * 0.76, 0.5);
          return <line key={index} x1={x} y1={y} x2={x} y2={y - 6 - level * 30} />;
        })}
      </g>
    );
  }

  // The model: a chip on the slab, its pins lit once the artifact hash is recorded.
  const chip = [onFace(cy, 0.3, 0.3), onFace(cy, 0.7, 0.3), onFace(cy, 0.7, 0.7), onFace(cy, 0.3, 0.7)];
  const pins = [0.38, 0.5, 0.62].flatMap((t) => [
    [onFace(cy, t, 0.3), onFace(cy, t, 0.18)],
    [onFace(cy, t, 0.7), onFace(cy, t, 0.82)],
    [onFace(cy, 0.3, t), onFace(cy, 0.18, t)],
    [onFace(cy, 0.7, t), onFace(cy, 0.82, t)],
  ]);
  return (
    <g>
      <g stroke={verified ? "var(--signal)" : "var(--ink)"} strokeWidth={2.5} strokeLinecap="round">
        {pins.map(([a, b], index) => (
          <line key={index} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} />
        ))}
      </g>
      <polygon points={points(chip)} fill="var(--ink)" stroke="var(--ink)" strokeWidth={1.5} />
    </g>
  );
}

function fingerprintFor(name: LayerName, lineage: Lineage | null): string | null {
  if (!lineage) {
    return null;
  }
  return {
    model: lineage.model.artifact_sha256,
    run: lineage.run.mlflow_run_id,
    config: lineage.config.fingerprint,
    dataset: lineage.dataset.fingerprint,
  }[name];
}

type LineageDrawingProps = {
  lineage: Lineage | null;
  exploded: boolean;
};

/**
 * An exploded technical drawing of one model's lineage: the dataset at the
 * base, then the configuration, the training run, and the model artifact.
 */
export function LineageDrawing({ lineage, exploded }: LineageDrawingProps) {
  const verified = Boolean(lineage?.model.artifact_sha256);

  return (
    <svg
      className="lineage"
      viewBox="0 0 640 610"
      role="img"
      aria-label="Exploded view of the model's lineage: dataset, configuration, training run, and model artifact"
      data-exploded={exploded}
    >
      {/* Painted bottom-up so each slab sits on the one below it. */}
      {[...LAYERS].reverse().map((layer) => {
        const position = LAYERS.indexOf(layer);
        const cy = EXPLODED_TOP + position * EXPLODED_GAP;
        const stackedOffset = STACKED_TOP + position * STACKED_GAP - cy;
        const anchor: Point = layer.side === "right" ? [CX + HALF_W, cy] : [CX - HALF_W, cy];
        // Callouts stay inside the 640-wide drawing so they never reach the spec sheet.
        const labelX = layer.side === "right" ? CX + HALF_W + 22 : CX - HALF_W - 22;
        const fingerprint = fingerprintFor(layer.name, lineage);

        return (
          <g
            key={layer.name}
            className="lineage__layer"
            style={{
              transform: exploded ? "translateY(0px)" : `translateY(${stackedOffset}px)`,
              transitionDelay: `${(LAYERS.length - position) * 90}ms`,
            }}
          >
            <Slab cy={cy} />
            <Motif name={layer.name} cy={cy} verified={verified} />

            <g className="lineage__callout">
              <line x1={anchor[0]} y1={anchor[1]} x2={labelX} y2={anchor[1]} stroke="var(--ink)" strokeWidth={1.5} />
              <circle cx={anchor[0]} cy={anchor[1]} r={4} fill="var(--ink)" />
              <circle cx={labelX} cy={anchor[1]} r={4} fill="var(--ink)" />
              <text
                x={layer.side === "right" ? labelX + 12 : labelX - 12}
                y={anchor[1] - 2}
                textAnchor={layer.side === "right" ? "start" : "end"}
                className="lineage__title"
              >
                {layer.title}
              </text>
              <text
                x={layer.side === "right" ? labelX + 12 : labelX - 12}
                y={anchor[1] + 16}
                textAnchor={layer.side === "right" ? "start" : "end"}
                className="lineage__hash"
              >
                {fingerprint ? shortHash(fingerprint, 8, 4) : "—"}
              </text>
            </g>
          </g>
        );
      })}
    </svg>
  );
}
