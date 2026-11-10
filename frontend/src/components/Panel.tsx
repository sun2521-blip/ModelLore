import type { ReactNode } from "react";

type PanelProps = {
  index: string;
  name: string;
  state?: ReactNode;
  id?: string;
  className?: string;
  children: ReactNode;
};

/**
 * A numbered console module, drawn like a sheet in a technical drawing: an ink
 * title tab on a top rule, a state key at the end of the rule, and corner
 * brackets instead of a closed box.
 */
export function Panel({ index, name, state, id, className, children }: PanelProps) {
  const titleId = `panel-title-${index}`;

  return (
    <section id={id} className={["panel", className].filter(Boolean).join(" ")} aria-labelledby={titleId}>
      <header className="panel__head">
        <h3 id={titleId} className="label panel__tab">
          <span className="mono panel__index">{index}</span> {name}
        </h3>
        <div className="panel__rule">{state ? <span className="label panel__state">{state}</span> : null}</div>
      </header>
      {children}
    </section>
  );
}
