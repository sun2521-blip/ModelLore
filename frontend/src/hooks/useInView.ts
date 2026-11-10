import { useCallback, useEffect, useState } from "react";

/**
 * True once the element has scrolled into view; used to start one-off reveals.
 * Returns a callback ref, so it works for elements that mount later (for
 * example once data has loaded), not only on the first render.
 */
export function useInView<T extends Element>(threshold = 0.25) {
  const [element, setElement] = useState<T | null>(null);
  const [inView, setInView] = useState(false);
  const ref = useCallback((node: T | null) => setElement(node), []);

  useEffect(() => {
    if (!element || inView) {
      return;
    }
    if (typeof IntersectionObserver === "undefined") {
      setInView(true);
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          setInView(true);
          observer.disconnect();
        }
      },
      { threshold },
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, [element, inView, threshold]);

  return { ref, inView };
}
