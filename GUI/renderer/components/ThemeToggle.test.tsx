import { render } from "preact";
import { afterEach, expect, it } from "vitest";
import { ThemeToggle } from "./ThemeToggle";

afterEach(() => {
  localStorage.clear();
  delete document.documentElement.dataset.theme;
});

it("switches between system, light and dark and shows the current choice", async () => {
  localStorage.setItem("metbit.theme", "light");
  const root = document.createElement("div");
  render(<ThemeToggle />, root);
  const button = (name: string) => root.querySelector<HTMLButtonElement>(`[aria-label="${name}"]`)!;
  expect(root.querySelector('[role="group"]')!.getAttribute("aria-label")).toBe("Colour theme");
  expect(button("Light theme").getAttribute("aria-pressed")).toBe("true");
  button("Dark theme").click();
  await new Promise((r) => setTimeout(r));
  expect(document.documentElement.dataset.theme).toBe("dark");
  expect(button("Dark theme").getAttribute("aria-pressed")).toBe("true");
  button("System theme").click();
  await new Promise((r) => setTimeout(r));
  expect(document.documentElement.dataset.theme).toBeUndefined();
});
