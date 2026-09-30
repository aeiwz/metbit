import { useEffect, useState } from "preact/hooks";

type Mode = "single" | "median";

export interface LabelFont {
  family?: string;
  weight?: string;
  style?: string;
}

const FAMILIES = ["Arial", "Helvetica", "Times New Roman", "Georgia", "Courier New", "Verdana"];
const STYLES: Record<string, Pick<LabelFont, "weight" | "style">> = {
  regular: {},
  bold: { weight: "bold" },
  italic: { style: "italic" },
  "bold-italic": { weight: "bold", style: "italic" },
};

interface Props {
  mode: Mode;
  onMode: (mode: Mode) => void;
  fontSize: number;
  onFontSize: (size: number) => void;
  font: LabelFont;
  onFont: (font: LabelFont) => void;
  peak: number | null;
}

// the toolbar of metbit's annotate_peak dashboard: spectra mode, label font size, current peak
export function AnnotateToolbar({ mode, onMode, fontSize, onFontSize, font, onFont, peak }: Props) {
  const styleName = `${font.weight === "bold" ? "bold" : ""}${font.weight === "bold" && font.style === "italic" ? "-" : ""}${font.style === "italic" ? "italic" : ""}` || "regular";
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
      <select
        class="toolbar-select"
        aria-label="Label font"
        value={font.family ?? ""}
        onChange={(e) => onFont({ ...font, family: e.currentTarget.value || undefined })}
      >
        <option value="">App font</option>
        {FAMILIES.map((f) => (
          <option key={f} value={f}>
            {f}
          </option>
        ))}
      </select>
      <select
        class="toolbar-select"
        aria-label="Label style"
        value={styleName}
        onChange={(e) => onFont({ family: font.family, ...STYLES[e.currentTarget.value] })}
      >
        <option value="regular">Regular</option>
        <option value="bold">Bold</option>
        <option value="italic">Italic</option>
        <option value="bold-italic">Bold italic</option>
      </select>
      <span class="peak-pill">Current peak: {peak === null ? "none" : `${peak.toFixed(3)} ppm`}</span>
    </div>
  );
}
