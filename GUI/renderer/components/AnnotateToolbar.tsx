import { useEffect, useState } from "preact/hooks";

type Mode = "single" | "median";

interface Props {
  mode: Mode;
  onMode: (mode: Mode) => void;
  fontSize: number;
  onFontSize: (size: number) => void;
  peak: number | null;
}

// the toolbar of metbit's annotate_peak dashboard: spectra mode, label font size, current peak
export function AnnotateToolbar({ mode, onMode, fontSize, onFontSize, peak }: Props) {
  const [size, setSize] = useState(fontSize); // follows the slider while dragging; saved on release
  useEffect(() => setSize(fontSize), [fontSize]);
  return (
    <div class="toolbar" role="toolbar" aria-label="Annotation tools">
      <div class="segmented" role="group" aria-label="Spectra">
        {(["single", "median"] as const).map((m) => (
          <button key={m} type="button" aria-pressed={mode === m} onClick={() => onMode(m)}>
            {m === "single" ? "Single spectra" : "Median spectra"}
          </button>
        ))}
      </div>
      <label class="toolbar-slider">
        Font size
        <input
          type="range"
          min={6}
          max={48}
          step={1}
          value={size}
          onInput={(e) => setSize(Number(e.currentTarget.value))}
          onChange={(e) => onFontSize(Number(e.currentTarget.value))}
        />
        <span class="toolbar-value">{size} px</span>
      </label>
      <span class="peak-pill">Current peak: {peak === null ? "none" : `${peak.toFixed(3)} ppm`}</span>
    </div>
  );
}
