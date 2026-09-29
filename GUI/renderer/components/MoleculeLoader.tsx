// Benzene ring in a 100 x 100 view box: six atoms on a hexagon, six sigma bonds, and six inner
// double-bond strokes in two alternating Kekule sets that cross-fade (resonance).
const CENTER = 50;
const RADIUS = 30;

const ATOMS = Array.from({ length: 6 }, (_, k) => {
  const a = (Math.PI / 3) * k - Math.PI / 2;
  return [CENTER + RADIUS * Math.cos(a), CENTER + RADIUS * Math.sin(a)] as const;
});

// inner stroke for bond k: pulled toward the center, trimmed at both ends
function innerBond(k: number): [number, number, number, number] {
  const pull = (v: number) => CENTER + (v - CENTER) * 0.76;
  const [x1, y1] = ATOMS[k];
  const [x2, y2] = ATOMS[(k + 1) % 6];
  const ax = pull(x1), ay = pull(y1), bx = pull(x2), by = pull(y2);
  const trim = 0.18;
  return [ax + (bx - ax) * trim, ay + (by - ay) * trim, bx - (bx - ax) * trim, by - (by - ay) * trim];
}

interface Props {
  caption: string;
  progress?: number | null;
}

export function MoleculeLoader({ caption, progress = null }: Props) {
  return (
    <div class="loader" role="status" aria-live="polite">
      <svg class="molecule" viewBox="0 0 100 100" aria-hidden="true">
        <g class="ring">
          {ATOMS.map(([x1, y1], k) => {
            const [x2, y2] = ATOMS[(k + 1) % 6];
            return <line key={`b${k}`} class="bond" x1={x1} y1={y1} x2={x2} y2={y2} />;
          })}
          {ATOMS.map((_, k) => {
            const [x1, y1, x2, y2] = innerBond(k);
            return <line key={`d${k}`} class={`double${k % 2 ? " alt" : ""}`} x1={x1} y1={y1} x2={x2} y2={y2} />;
          })}
          {ATOMS.map(([cx, cy], k) => (
            <circle key={`a${k}`} class="atom" cx={cx} cy={cy} r={4.5} style={{ animationDelay: `${k * 0.25}s` }} />
          ))}
        </g>
      </svg>
      <span class="loader-caption">{caption}</span>
      {typeof progress === "number" && <span class="loader-progress">{Math.round(progress * 100)}%</span>}
    </div>
  );
}
