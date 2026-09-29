import type { Annotation } from "../api";

// labels are stored HTML-escaped (for Plotly); show the plain text
const decode = (s: string) => new DOMParser().parseFromString(s, "text/html").documentElement.textContent ?? s;

interface Props {
  items: Annotation[];
  onDelete: (id: string) => void;
  onImport: () => void;
  onExport: () => void;
}

export function AnnotationList({ items, onDelete, onImport, onExport }: Props) {
  return (
    <div class="annotations">
      <h3>Labels</h3>
      {items.length === 0 ? (
        <p class="hint">Click a peak on the spectra to add a label.</p>
      ) : (
        <ul class="label-list">
          {items.map((a) => (
            <li key={a.id}>
              <span>{decode(a.label)}</span>
              <span class="ppm-tag">{a.ppm.toFixed(3)} ppm</span>
              <button type="button" class="link" onClick={() => onDelete(a.id)}>
                Delete
              </button>
            </li>
          ))}
        </ul>
      )}
      <div class="row">
        <button type="button" class="link" onClick={onImport}>
          Import labels…
        </button>
        <button type="button" class="link" onClick={onExport} disabled={items.length === 0}>
          Export labels…
        </button>
      </div>
    </div>
  );
}
