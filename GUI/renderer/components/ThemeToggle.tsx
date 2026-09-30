import { useState } from "preact/hooks";
import { applyTheme, initialTheme, type ThemePref } from "../themePref";

// 16 px line icons, stroked in currentColor
const ICONS: Record<ThemePref, preact.JSX.Element> = {
  system: (
    <>
      <rect x="2" y="3" width="12" height="8" rx="1.5" />
      <path d="M6 14h4M8 11v3" />
    </>
  ),
  light: (
    <>
      <circle cx="8" cy="8" r="3" />
      <path d="M8 1.5v1.5M8 13v1.5M1.5 8H3M13 8h1.5M3.4 3.4l1 1M11.6 11.6l1 1M3.4 12.6l1-1M11.6 4.4l1-1" />
    </>
  ),
  dark: <path d="M13.5 9.5A5.5 5.5 0 0 1 6.5 2.5a5.5 5.5 0 1 0 7 7z" />,
};

const LABELS: Record<ThemePref, string> = { system: "System theme", light: "Light theme", dark: "Dark theme" };

export function ThemeToggle() {
  const [pref, setPref] = useState<ThemePref>(initialTheme);
  return (
    <div class="theme-toggle" role="group" aria-label="Colour theme">
      {(["system", "light", "dark"] as const).map((p) => (
        <button
          key={p}
          type="button"
          aria-label={LABELS[p]}
          title={LABELS[p]}
          aria-pressed={pref === p}
          onClick={() => {
            applyTheme(p);
            setPref(p);
          }}
        >
          <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
            {ICONS[p]}
          </svg>
        </button>
      ))}
    </div>
  );
}
