import { Fragment } from "preact";
import type { StageInfo } from "../api";

interface Props {
  stages: StageInfo[];
  current: string;
  onSelect: (id: string) => void;
}

export function Rail({ stages, current, onSelect }: Props) {
  const statusOf = Object.fromEntries(stages.map((s) => [s.id, s.status]));
  return (
    <nav class="rail" aria-label="Pipeline">
      <div class="rail-brand">metbit</div>
      {stages.map((s, i) => {
        // reachable when its input exists, or it already has its own result
        const reachable = !s.requires || statusOf[s.requires] === "done" || s.status !== "empty";
        return (
          <Fragment key={s.id}>
            {s.section !== stages[i - 1]?.section && <div class="rail-section">{s.section}</div>}
            <button
              aria-current={s.id === current}
              disabled={!reachable || !!s.blocked}
              title={s.blocked ?? undefined}
              onClick={() => onSelect(s.id)}
            >
              <span class={`dot ${s.status}`} />
              {s.title}
            </button>
          </Fragment>
        );
      })}
    </nav>
  );
}
