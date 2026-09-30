export type ThemePref = "system" | "light" | "dark";

export const STORAGE_KEY = "metbit.theme";
const EVENT = "metbit-theme";
const PREFS: ThemePref[] = ["system", "light", "dark"];

export function initialTheme(): ThemePref {
  try {
    const saved = localStorage.getItem(STORAGE_KEY) as ThemePref | null;
    return saved && PREFS.includes(saved) ? saved : "system";
  } catch {
    return "system";
  }
}

/** Light or dark forces the palette (data-theme on <html>); system leaves it to prefers-color-scheme. */
export function applyTheme(pref: ThemePref) {
  const root = document.documentElement;
  if (pref === "system") delete root.dataset.theme;
  else root.dataset.theme = pref;
  try {
    localStorage.setItem(STORAGE_KEY, pref);
  } catch {
    // ponytail: storage blocked; the choice still applies for this session
  }
  window.metbit?.setTheme?.(pref); // native menus, scrollbars and the window frame follow too
  window.dispatchEvent(new Event(EVENT));
}

/** Called when the palette may have changed: a new choice, or the OS switching while on "system". */
export function onThemeChange(fn: () => void): () => void {
  const media = window.matchMedia?.("(prefers-color-scheme: dark)");
  window.addEventListener(EVENT, fn);
  media?.addEventListener?.("change", fn);
  return () => {
    window.removeEventListener(EVENT, fn);
    media?.removeEventListener?.("change", fn);
  };
}
