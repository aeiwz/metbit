import { afterEach, expect, it, vi } from "vitest";
import { applyTheme, initialTheme, onThemeChange, STORAGE_KEY } from "./themePref";

afterEach(() => {
  localStorage.clear();
  delete document.documentElement.dataset.theme;
});

it("sets the root theme attribute, remembers the choice and tells the app shell", () => {
  const setTheme = vi.fn();
  (window as any).metbit = { setTheme };
  applyTheme("dark");
  expect(document.documentElement.dataset.theme).toBe("dark");
  expect(localStorage.getItem(STORAGE_KEY)).toBe("dark");
  expect(setTheme).toHaveBeenCalledWith("dark");
  applyTheme("system"); // follow the OS: no attribute, so the media query decides
  expect(document.documentElement.dataset.theme).toBeUndefined();
  expect(initialTheme()).toBe("system");
});

it("starts from the saved choice and ignores junk", () => {
  localStorage.setItem(STORAGE_KEY, "light");
  expect(initialTheme()).toBe("light");
  localStorage.setItem(STORAGE_KEY, "purple");
  expect(initialTheme()).toBe("system");
});

it("notifies listeners (plots re-read colours) on every change", () => {
  const seen = vi.fn();
  const stop = onThemeChange(seen);
  applyTheme("light");
  applyTheme("dark");
  stop();
  applyTheme("system");
  expect(seen).toHaveBeenCalledTimes(2);
});

it("still works when storage is unavailable", () => {
  const spy = vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
    throw new Error("blocked");
  });
  expect(() => applyTheme("dark")).not.toThrow();
  spy.mockRestore();
});
