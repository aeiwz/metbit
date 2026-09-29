import { render } from "preact";
import { expect, it, vi } from "vitest";
import { StocsyTools } from "./StocsyTools";

function mount(recent: { anchor_ppm: number; p_threshold: number }[]) {
  const root = document.createElement("div");
  const onUse = vi.fn();
  const onClear = vi.fn();
  render(<StocsyTools recent={recent} onUse={onUse} onClear={onClear} />, root);
  return { root, onUse, onClear };
}

const button = (root: HTMLElement, text: string) =>
  [...root.querySelectorAll("button")].find((b) => b.textContent === text) as HTMLButtonElement;

it("lists recent anchors and reuses one", () => {
  const { root, onUse } = mount([{ anchor_ppm: 3.03, p_threshold: 0.0001 }]);
  button(root, "3.030 ppm").click();
  expect(onUse).toHaveBeenCalledWith({ anchor_ppm: 3.03, p_threshold: 0.0001 });
});

it("clears the anchor", () => {
  const { root, onClear } = mount([]);
  button(root, "Clear anchor").click();
  expect(onClear).toHaveBeenCalledOnce();
});

it("explains the workflow when nothing has run yet", () => {
  expect(mount([]).root.textContent).toContain("Click a peak on the spectra, then Run.");
});
