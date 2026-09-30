// metbit identity (GUI/metbit-film/identity): the "resolved m", a baseline rising through two rounded
// resonances, plus a small square for one sampled variable (the "bit"). Geometry copied from the
// identity's generate_logo.py; do not stretch, recolour per part or replace the peaks.
export const MARK_PATH = "M 12 92 H 27 C 43 92 35 26 52 26 C 69 26 61 92 76 92 C 91 92 82 42 99 42 C 116 42 107 92 125 92 H 143";

export function BrandMark({ size = 32 }: { size?: number }) {
  return (
    <svg class="brand-mark" width={size} height={(size * 120) / 172} viewBox="0 0 172 120" aria-hidden="true">
      <path d={MARK_PATH} fill="none" stroke-width="5.5" stroke-linecap="round" stroke-linejoin="round" />
      {size >= 32 && <rect x="150" y="84" width="8" height="8" />}
    </svg>
  );
}

// the name stays lowercase; the GUI lockup adds a quiet descriptor
export function Lockup({ gui = false, size = 32 }: { gui?: boolean; size?: number }) {
  return (
    <span class="lockup">
      <BrandMark size={size} />
      <span class="lockup-name">metbit</span>
      {gui && <span class="lockup-gui">GUI</span>}
    </span>
  );
}
