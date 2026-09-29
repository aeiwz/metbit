import { render } from "preact";
import { expect, it, vi } from "vitest";
import type { PlotStyle } from "../api";
import { StyleForm, applyPatch } from "./StyleForm";

const STYLE: PlotStyle = {
  groups: { A: { color: "#112233", symbol: "circle" }, B: { color: "#445566", symbol: "square" } },
  marker_size: 12,
  marker_opacity: 0.8,
  font_size: 14,
  line_width: 1.2,
  show_ellipses: true,
};

function mount(props: Partial<Parameters<typeof StyleForm>[0]> = {}) {
  const root = document.createElement("div");
  const onChange = vi.fn();
  render(<StyleForm effective={STYLE} onChange={onChange} {...props} />, root);
  return { root, onChange };
}

const change = (el: HTMLInputElement | HTMLSelectElement, value: string) => {
  el.value = value;
  el.dispatchEvent(new Event("change"));
};

it("renders a row per group and reports color and symbol changes", () => {
  const { root, onChange } = mount();
  expect(root.querySelectorAll(".style-group")).toHaveLength(2);
  expect(root.querySelector<HTMLInputElement>('[aria-label="A color"]')!.value).toBe("#112233");
  change(root.querySelector<HTMLSelectElement>('[aria-label="B symbol"]')!, "diamond");
  expect(onChange).toHaveBeenLastCalledWith({ groups: { B: { symbol: "diamond" } } });
  change(root.querySelector<HTMLInputElement>('[aria-label="A color"]')!, "#abcdef");
  expect(onChange).toHaveBeenLastCalledWith({ groups: { A: { color: "#abcdef" } } });
});

it("clamps numbers to their range", () => {
  const { root, onChange } = mount();
  change(root.querySelector<HTMLInputElement>("#style-marker_size")!, "100");
  expect(onChange).toHaveBeenLastCalledWith({ marker_size: 60 });
});

it("toggles ellipses", () => {
  const { root, onChange } = mount();
  root.querySelector<HTMLInputElement>('input[type="checkbox"]')!.click();
  expect(onChange).toHaveBeenLastCalledWith({ show_ellipses: false });
});

it("marks overridden fields and offers a reset", () => {
  const onReset = vi.fn();
  const { root } = mount({ override: { marker_size: 30, groups: { A: { color: "#000000" } } }, onReset });
  expect(root.querySelectorAll(".style-group.overridden")).toHaveLength(1);
  expect(root.querySelector("#style-marker_size")!.closest(".field")!.classList.contains("overridden")).toBe(true);
  [...root.querySelectorAll("button")].find((b) => b.textContent === "Use global style")!.click();
  expect(onReset).toHaveBeenCalledOnce();
});

it("asks for data when there are no groups", () => {
  const { root } = mount({ effective: { ...STYLE, groups: {} } });
  expect(root.textContent).toContain("Import data to style its groups.");
});

it("applyPatch merges groups deeply", () => {
  const next = applyPatch(STYLE, { groups: { A: { symbol: "x" } }, font_size: 20 });
  expect(next.groups.A).toEqual({ color: "#112233", symbol: "x" });
  expect(next.groups.B).toEqual(STYLE.groups.B);
  expect(next.font_size).toBe(20);
});
