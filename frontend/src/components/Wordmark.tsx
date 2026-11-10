const TRACK_NODES = [
  { x: 48, label: "DATA" },
  { x: 256, label: "CONFIG" },
  { x: 464, label: "RUN" },
  { x: 672, label: "MODEL" },
];

type WordmarkProps = {
  /** "hero" adds the lineage track underneath; "tab" is the compact header mark. */
  variant?: "hero" | "tab";
};

/**
 * The ModelLore wordmark: heavy extended italic type with an offset olive
 * shadow, riding on a lineage track (data, config, run, model).
 */
export function Wordmark({ variant = "hero" }: WordmarkProps) {
  const isHero = variant === "hero";

  return (
    <svg
      className={`wordmark wordmark--${variant}`}
      viewBox={isHero ? "0 0 720 182" : "0 0 720 124"}
      role="img"
      aria-label="ModelLore"
    >
      <g
        fontFamily="'Archivo Variable', Archivo, sans-serif"
        fontWeight={900}
        fontStyle="italic"
        fontSize={104}
        style={{ fontStretch: "125%" }}
      >
        <text
          x={368}
          y={110}
          textAnchor="middle"
          textLength={652}
          lengthAdjust="spacingAndGlyphs"
          fill="var(--olive-deep)"
          stroke="var(--ink)"
          strokeWidth={3}
          strokeLinejoin="round"
        >
          MODELLORE
        </text>
        <text
          x={360}
          y={102}
          textAnchor="middle"
          textLength={652}
          lengthAdjust="spacingAndGlyphs"
          fill="var(--ink)"
        >
          MODELLORE
        </text>
      </g>

      {isHero && (
        <g className="wordmark__track">
          <line
            x1={TRACK_NODES[0].x}
            y1={142}
            x2={TRACK_NODES[TRACK_NODES.length - 1].x}
            y2={142}
            stroke="var(--ink)"
            strokeWidth={3}
            strokeLinecap="round"
          />
          {TRACK_NODES.map((node, index) => (
            <g key={node.label}>
              <circle
                cx={node.x}
                cy={142}
                r={index === TRACK_NODES.length - 1 ? 9 : 7}
                fill={index === TRACK_NODES.length - 1 ? "var(--sage)" : "var(--ink)"}
                stroke="var(--ink)"
                strokeWidth={3}
              />
              <text
                x={node.x}
                y={174}
                textAnchor="middle"
                fontFamily="'JetBrains Mono', monospace"
                fontSize={15}
                fontWeight={500}
                letterSpacing="0.08em"
                fill="var(--ink)"
              >
                {node.label}
              </text>
            </g>
          ))}
        </g>
      )}
    </svg>
  );
}
