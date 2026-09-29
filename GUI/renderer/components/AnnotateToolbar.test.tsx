import { render } from "preact";
import { expect, it, vi } from "vitest";
import { AnnotateToolbar } from "./AnnotateToolbar";

function mount(peak: number | null = 3.0251) {
  const root = document.createElement("div");
  const onMode = vi.fn();
  const onFontSize = vi.fn();
  render(<AnnotateToolbar mode="single" onMode={onMode} fontSize={14} onFontSize={onFontSize} peak={peak} />, root);
  return { root, onMode, onFontSize };
}

it("switches mode and shows the current peak", () => {
  const { root, onMode } = mount();
  const button = (t: string) => [...root.querySelectorAll("button")].find((b) => b.textContent === t)!;
  expect(button("Single spectra").getAttribute("aria-pressed")).toBe("true");
  button("Median spectra").click();
  expect(onMode).toHaveBeenCalledWith("median");
  expect(root.querySelector(".peak-pill")!.textContent).toBe("Current peak: 3.025 ppm");
  expect(mount(null).root.querySelector(".peak-pill")!.textContent).toBe("Current peak: none");
});

it("reports the label font size when the slider is released", () => {
  const { root, onFontSize } = mount();
  const slider = root.querySelector<HTMLInputElement>('input[type="range"]')!;
  slider.value = "22";
  slider.dispatchEvent(new Event("change"));
  expect(onFontSize).toHaveBeenCalledWith(22);
});
