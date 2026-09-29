export interface Anchor {
  anchor_ppm: number;
  p_threshold: number;
}

interface Props {
  recent: Anchor[];
  onUse: (anchor: Anchor) => void;
  onClear: () => void;
}

export function StocsyTools({ recent, onUse, onClear }: Props) {
  return (
    <div class="annotations stocsy-tools">
      <div class="row">
        <button type="button" class="link" onClick={onClear}>
          Clear anchor
        </button>
      </div>
      <h3>Recent anchors</h3>
      {recent.length === 0 ? (
        <p class="hint">Click a peak on the spectra, then Run.</p>
      ) : (
        <ul class="label-list">
          {recent.map((a) => (
            <li key={`${a.anchor_ppm}:${a.p_threshold}`}>
              <button type="button" class="link" onClick={() => onUse(a)}>
                {a.anchor_ppm.toFixed(3)} ppm
              </button>
              <span class="ppm-tag">p {a.p_threshold}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
