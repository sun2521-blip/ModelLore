import type { ReactNode } from "react";

type PanelProps = {
  index: string;
  name: string;
  state?: ReactNode;
  id?: string;
  className?: string;
  children: ReactNode;
};

/** A numbered console module: header strip with index, name, and a state badge. */
export function Panel({ index, name, state, id, className, children }: PanelProps) {
  const titleId = `panel-title-${index}`;

  return (
    <section id={id} className={["panel", className].filter(Boolean).join(" ")} aria-labelledby={titleId}>
      <header className="panel__head">
        <span className="mono panel__index">{index}</span>
        <h3 id={titleId} className="label panel__name">
          {name}
        </h3>
        {state ? <span className="label panel__state">{state}</span> : <span />}
      </header>
      {children}
    </section>
  );
}
