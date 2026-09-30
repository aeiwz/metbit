import type { PlotStyle, StyleOverride } from "../api";
import { DecimalInput } from "./DecimalInput";

export const SYMBOLS = ["circle", "square", "diamond", "triangle-up", "triangle-down", "cross", "x", "star"];

type NumberKey = "marker_size" | "marker_opacity" | "font_size" | "line_width";
const NUMBERS: { key: NumberKey; label: string; min: number; max: number; step: number }[] = [
  { key: "marker_size", label: "Marker size", min: 4, max: 60, step: 1 },
  { key: "marker_opacity", label: "Marker opacity", min: 0.1, max: 1, step: 0.05 },
  { key: "font_size", label: "Font size", min: 8, max: 32, step: 1 },
  { key: "line_width", label: "Line width", min: 0.5, max: 5, step: 0.1 },
];

const clamp = (v: number, min: number, max: number) => Math.min(max, Math.max(min, v));

export function applyPatch<T extends StyleOverride>(base: T, patch: StyleOverride): T {
  const groups: Record<string, any> = { ...(base.groups ?? {}) };
  for (const [name, g] of Object.entries(patch.groups ?? {})) groups[name] = { ...(groups[name] ?? {}), ...g };
  return { ...base, ...patch, groups };
}

interface Props {
  effective: PlotStyle;
  override?: StyleOverride;
  onChange: (patch: StyleOverride) => void;
  onReset?: () => void;
}

export function StyleForm({ effective, override, onChange, onReset }: Props) {
  const mark = (on: boolean) => (override && on ? " overridden" : "");
  const names = Object.keys(effective.groups);
  const customized =
    override && Object.entries(override).some(([k, v]) => (k === "groups" ? Object.keys(v ?? {}).length > 0 : v !== undefined));
  return (
    <div class="style-form">
      {names.length === 0 ? (
        <p class="hint">Import data to style its groups.</p>
      ) : (
        <div class="style-groups">
          {names.map((name) => {
            const g = effective.groups[name];
            return (
              <div key={name} class={`style-group${mark(!!override?.groups?.[name])}`}>
                <input
                  type="color"
                  aria-label={`${name} color`}
                  value={g.color}
                  onChange={(e) => onChange({ groups: { [name]: { color: e.currentTarget.value } } })}
                />
                <span class="style-name" title={name}>
                  {name}
                </span>
                <select
                  aria-label={`${name} symbol`}
                  value={g.symbol}
                  onChange={(e) => onChange({ groups: { [name]: { symbol: e.currentTarget.value } } })}
                >
                  {SYMBOLS.map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </div>
            );
          })}
        </div>
      )}
      {NUMBERS.map((f) => (
        <div key={f.key} class={`field${mark(override?.[f.key] !== undefined)}`}>
          <label for={`style-${f.key}`}>{f.label}</label>
          <DecimalInput
            id={`style-${f.key}`}
            value={effective[f.key]}
            onValue={(v) => {
              if (typeof v === "number") onChange({ [f.key]: clamp(v, f.min, f.max) });
            }}
          />
          <span class="hint">
            {f.min} to {f.max}
          </span>
        </div>
      ))}
      <label class={`check${mark(override?.show_ellipses !== undefined)}`}>
        <input
          type="checkbox"
          checked={effective.show_ellipses}
          onClick={(e) => onChange({ show_ellipses: e.currentTarget.checked })}
        />
        Ellipse per group
      </label>
      <span class="hint">Off draws one overall Hotelling ellipse (metbit's individual_ellipse).</span>
      {onReset && customized && (
        <button type="button" class="link" onClick={onReset}>
          Use global style
        </button>
      )}
    </div>
  );
}
