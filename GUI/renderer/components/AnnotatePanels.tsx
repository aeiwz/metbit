import type { ComponentChildren } from "preact";
import { useState } from "preact/hooks";

export interface LineStyleValue {
  color?: string;
  dash?: string;
  width?: number;
}

interface Props {
  peak: number | null;
  onAdd: (label: string, angle: number) => void;
  targets: { groups: string[]; samples: string[] };
  lineStyles: Record<string, LineStyleValue>;
  onApplyStyle: (target: string, style: LineStyleValue) => void;
  onResetStyle: (target: string) => void;
  exports: ComponentChildren;
  children: ComponentChildren; // the current labels list
}

const DASHES = ["solid", "dash", "dot", "dashdot"];

// the collapsible panels of metbit's annotate_peak dashboard
export function AnnotatePanels({ peak, onAdd, targets, lineStyles, onApplyStyle, onResetStyle, exports, children }: Props) {
  const [label, setLabel] = useState("");
  const [angle, setAngle] = useState(0);
  const [target, setTarget] = useState("");
  const [color, setColor] = useState("#2563eb");
  const [dash, setDash] = useState("solid");
  const [width, setWidth] = useState(1.5);
  const chosen = target || targets.groups[0] || "";
  const canAdd = peak !== null && label.trim() !== "";
  return (
    <div class="annotate-panels">
      <details class="dash-panel" open>
        <summary>📍 Add annotation</summary>
        <form
          class="dash-panel-body"
          onSubmit={(e) => {
            e.preventDefault();
            if (!canAdd) return;
            onAdd(label.trim(), angle);
            setLabel("");
          }}
        >
          <span class="hint">{peak === null ? "Click a peak on the spectra." : `Current peak: ${peak.toFixed(3)} ppm`}</span>
          <label class="field-inline">
            Label
            <input id="annotate-label" placeholder="Enter peak label" value={label} onInput={(e) => setLabel(e.currentTarget.value)} />
          </label>
          <label class="field-inline">
            Angle (°)
            <input
              type="number"
              min={-180}
              max={180}
              step={5}
              value={angle}
              onInput={(e) => setAngle(Number(e.currentTarget.value) || 0)}
            />
          </label>
          <button class="btn primary" type="submit" disabled={!canAdd}>
            Add Label
          </button>
        </form>
      </details>
      <details class="dash-panel">
        <summary>🎨 Style line</summary>
        <div class="dash-panel-body">
          <select aria-label="Line to style" value={chosen} onChange={(e) => setTarget(e.currentTarget.value)}>
            <optgroup label="Groups">
              {targets.groups.map((g) => (
                <option key={g} value={g}>
                  {g}
                </option>
              ))}
            </optgroup>
            <optgroup label="Samples">
              {targets.samples.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </optgroup>
          </select>
          <div class="row">
            <input type="color" aria-label="Line color" value={color} onChange={(e) => setColor(e.currentTarget.value)} />
            <select aria-label="Line dash" value={dash} onChange={(e) => setDash(e.currentTarget.value)}>
              {DASHES.map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>
            <input
              type="number"
              aria-label="Line width"
              min={0.1}
              max={10}
              step={0.1}
              value={width}
              onChange={(e) => setWidth(Number(e.currentTarget.value) || 1.5)}
            />
          </div>
          <div class="row">
            <button class="btn" type="button" disabled={!chosen} onClick={() => onApplyStyle(chosen, { color, dash, width })}>
              Apply Style
            </button>
            {lineStyles[chosen] && (
              <button class="link" type="button" onClick={() => onResetStyle(chosen)}>
                Reset
              </button>
            )}
          </div>
        </div>
      </details>
      <details class="dash-panel" open>
        <summary>📌 Current labels</summary>
        <div class="dash-panel-body">{children}</div>
      </details>
      <div class="row dash-exports">{exports}</div>
    </div>
  );
}
