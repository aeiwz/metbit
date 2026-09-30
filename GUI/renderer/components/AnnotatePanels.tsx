import type { ComponentChildren } from "preact";
import { useState } from "preact/hooks";
import type { Annotation } from "../api";
import { decode } from "./AnnotationList";
import { DecimalInput } from "./DecimalInput";

export interface LineStyleValue {
  color?: string;
  dash?: string;
  width?: number;
}

interface Props {
  peak: number | null;
  onAdd: (label: string, angle: number) => void;
  editing: Annotation | null; // a label picked on the plot or in the list
  onSave: (id: string, label: string, angle: number) => void;
  onCancelEdit: () => void;
  targets: { groups: string[]; samples: string[] };
  lineStyles: Record<string, LineStyleValue>;
  onApplyStyle: (target: string, style: LineStyleValue) => void;
  onResetStyle: (target: string) => void;
  exports: ComponentChildren;
  children: ComponentChildren; // the current labels list
}

type FormProps = Pick<Props, "peak" | "editing" | "onAdd" | "onSave" | "onCancelEdit">;

function LabelForm({ peak, editing, onAdd, onSave, onCancelEdit }: FormProps) {
  const [label, setLabel] = useState(editing ? decode(editing.label) : "");
  const [angle, setAngle] = useState<number | string>(editing ? editing.angle : 0);
  const angleOk = typeof angle === "number" && angle >= -180 && angle <= 180;
  const canAdd = (editing !== null || peak !== null) && label.trim() !== "" && angleOk;
  return (
    <form
      class="dash-panel-body"
      onSubmit={(e) => {
        e.preventDefault();
        if (!canAdd) return;
        if (editing) {
          onSave(editing.id, label.trim(), angle as number);
          return;
        }
        onAdd(label.trim(), angle as number);
        setLabel("");
      }}
    >
      <span class="hint">
        {editing
          ? `Label at ${editing.ppm.toFixed(3)} ppm`
          : peak === null
            ? "Click a peak on the spectra, or a label to edit it."
            : `Current peak: ${peak.toFixed(3)} ppm`}
      </span>
      <label class="field-inline">
        Label
        <input id="annotate-label" placeholder="Enter peak label" value={label} onInput={(e) => setLabel(e.currentTarget.value)} />
      </label>
      <label class="field-inline">
        Angle (°)
        <DecimalInput aria-label="Angle" value={angle} live onValue={setAngle} />
      </label>
      {!angleOk && <span class="err">Angle must be a number from -180 to 180</span>}
      <div class="row">
        <button class="btn primary" type="submit" disabled={!canAdd}>
          {editing ? "Save" : "Add Label"}
        </button>
        {editing && (
          <button class="link" type="button" onClick={onCancelEdit}>
            Cancel
          </button>
        )}
      </div>
    </form>
  );
}

const DASHES = ["solid", "dash", "dot", "dashdot"];

// the collapsible panels of metbit's annotate_peak dashboard
export function AnnotatePanels({ peak, onAdd, editing, onSave, onCancelEdit, targets, lineStyles, onApplyStyle, onResetStyle, exports, children }: Props) {
  const [target, setTarget] = useState("");
  const [color, setColor] = useState("#2563eb");
  const [dash, setDash] = useState("solid");
  const [width, setWidth] = useState(1.5);
  const chosen = target || targets.groups[0] || "";
  return (
    <div class="annotate-panels">
      <details class="dash-panel" open>
        <summary>{editing ? "✏️ Edit annotation" : "📍 Add annotation"}</summary>
        {/* keyed: picking another label starts the form from that label's text and angle */}
        <LabelForm key={editing?.id ?? "new"} peak={peak} editing={editing} onAdd={onAdd} onSave={onSave} onCancelEdit={onCancelEdit} />
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
            <DecimalInput
              aria-label="Line width"
              value={width}
              onValue={(v) => typeof v === "number" && setWidth(Math.min(10, Math.max(0.1, v)))}
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
