export function JobBar({ pct }: { pct: number | null }) {
  return (
    <div class="jobbar" role="progressbar" aria-valuenow={pct == null ? undefined : Math.round(pct * 100)}>
      <div style={{ width: `${(pct ?? 0) * 100}%`, opacity: pct == null ? 0 : 1 }} />
    </div>
  );
}
