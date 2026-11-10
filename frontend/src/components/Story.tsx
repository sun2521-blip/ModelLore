import { useInView } from "../hooks/useInView";

const MARQUEE_WORDS = ["Train it", "Track it", "Trace it"];

/** A slow tape of giant type between the hero and the console. */
export function Marquee() {
  const sequence = [...MARQUEE_WORDS, ...MARQUEE_WORDS];

  return (
    <section className="marquee" aria-label="Train it. Track it. Trace it.">
      <div className="marquee__track" aria-hidden="true">
        {[0, 1].map((copy) => (
          <span key={copy} className="marquee__run display">
            {sequence.map((word, index) => (
              <span key={`${copy}-${index}`} className="marquee__word">
                {word}
                <span className="marquee__dot" />
              </span>
            ))}
          </span>
        ))}
      </div>
    </section>
  );
}

const REPOSITORY_URL = "https://github.com/sun2521-blip/ModelLore";

/** The closing signal: the promise, and where to read more. */
export function Finale() {
  const { ref, inView } = useInView<HTMLElement>(0.2);

  return (
    <section className="finale" ref={ref} data-in-view={inView} aria-labelledby="finale-title">
      <p className="label finale__kicker">End of tape</p>
      <h2 id="finale-title" className="display finale__title">
        Know how your
        <br />
        models came to be
      </h2>
      <svg className="finale__cord" viewBox="0 0 420 60" aria-hidden="true">
        <path
          d="M0 30 H120 c10 0 10 -22 20 -22 s10 44 20 44 s10 -44 20 -44 s10 44 20 44 s10 -44 20 -44 s10 44 20 44 s10 -22 20 -22 H420"
          fill="none"
          stroke="var(--ink)"
          strokeWidth={4}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <circle cx="414" cy="30" r="7" fill="var(--ink)" />
      </svg>
      <div className="finale__links">
        <a className="button button--ink label" href={REPOSITORY_URL} target="_blank" rel="noreferrer noopener">
          Read the source
        </a>
        <a className="button button--ink label" href="/docs" target="_blank" rel="noreferrer noopener">
          API reference
        </a>
      </div>
    </section>
  );
}
