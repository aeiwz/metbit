import { useState } from "preact/hooks";

interface Props {
  ppm: number;
  x: number; // viewport coordinates of the click
  y: number;
  onAdd: (label: string, angle: number) => void;
  onCancel: () => void;
}

export function AnnotationPopover({ ppm, x, y, onAdd, onCancel }: Props) {
  const [label, setLabel] = useState("");
  const [angle, setAngle] = useState(0);
  return (
    <form
      class="popover"
      style={{ left: `${x}px`, top: `${y}px` }}
      onSubmit={(e) => {
        e.preventDefault();
        if (label.trim()) onAdd(label.trim(), angle);
      }}
      onKeyDown={(e) => {
        if (e.key === "Escape") onCancel();
      }}
    >
      <span class="popover-title">Label at {ppm.toFixed(3)} ppm</span>
      <input aria-label="Label" autoFocus placeholder="e.g. lactate" value={label} onInput={(e) => setLabel(e.currentTarget.value)} />
      <input
        aria-label="Angle"
        type="number"
        min={-180}
        max={180}
        step={5}
        value={angle}
        onInput={(e) => setAngle(Number(e.currentTarget.value) || 0)}
      />
      <div class="row">
        <button class="btn primary" type="submit" disabled={!label.trim()}>
          Add
        </button>
        <button class="btn" type="button" onClick={onCancel}>
          Cancel
        </button>
      </div>
    </form>
  );
}
