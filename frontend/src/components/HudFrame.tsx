import { useEffect, useState } from "react";

// The header and footer own the top and bottom edges, so each corner carries a
// single readout on its vertical arm: the window's width or height.
const CORNERS = [
  { corner: "tl", axis: "W" },
  { corner: "tr", axis: "H" },
  { corner: "bl", axis: "H" },
  { corner: "br", axis: "W" },
] as const;

function useViewportSize() {
  const read = () => ({ width: window.innerWidth, height: window.innerHeight });
  const [size, setSize] = useState(read);

  useEffect(() => {
    const onResize = () => setSize(read());
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  return size;
}

/** A flared edge marker, the notch at the middle of each side of the frame. */
function Flare({ side }: { side: "left" | "right" }) {
  return (
    <svg className={`hud__flare hud__flare--${side}`} viewBox="0 0 22 90" aria-hidden="true">
      <path
        d={
          side === "left"
            ? "M0 0 C0 30 4 40 22 45 C4 50 0 60 0 90 Z"
            : "M22 0 C22 30 18 40 0 45 C18 50 22 60 22 90 Z"
        }
        fill="var(--ink)"
      />
    </svg>
  );
}

/**
 * The instrument frame drawn around the viewport: corner brackets annotated
 * with the window's real dimensions, like a technical drawing of the page.
 */
export function HudFrame() {
  const { width, height } = useViewportSize();

  return (
    <div className="hud" aria-hidden="true">
      {CORNERS.map(({ corner, axis }) => (
        <div key={corner} className={`hud__corner hud__corner--${corner}`}>
          <span className="hud__pill" />
          <span className="hud__dim mono">
            {axis} {axis === "W" ? width : height}
          </span>
        </div>
      ))}
      <Flare side="left" />
      <Flare side="right" />
    </div>
  );
}
