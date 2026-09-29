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
      {stages.map((s) => {
        // reachable when its input exists, or it already has its own result
        const reachable = !s.requires || statusOf[s.requires] === "done" || s.status !== "empty";
        return (
          <button key={s.id} aria-current={s.id === current} disabled={!reachable} onClick={() => onSelect(s.id)}>
            <span class={`dot ${s.status}`} title={s.status} />
            {s.title}
          </button>
        );
      })}
    </nav>
  );
}
