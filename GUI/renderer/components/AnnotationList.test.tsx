import { render } from "preact";
import { expect, it, vi } from "vitest";
import type { Annotation } from "../api";
import { AnnotationList } from "./AnnotationList";

function mount(items: Annotation[]) {
  const root = document.createElement("div");
  const handlers = { onDelete: vi.fn(), onImport: vi.fn(), onExport: vi.fn(), onEdit: vi.fn() };
  render(<AnnotationList items={items} {...handlers} />, root);
  return { root, ...handlers };
}

const button = (root: HTMLElement, text: string) =>
  [...root.querySelectorAll("button")].find((b) => b.textContent === text) as HTMLButtonElement;

it("lists decoded labels with their ppm and deletes by id", () => {
  const { root, onDelete } = mount([{ id: "a1", ppm: 1.3312, label: "&lt;b&gt;Lac", angle: 0 }]);
  expect(root.querySelector(".label-list li")!.textContent).toContain("<b>Lac");
  expect(root.textContent).toContain("1.331 ppm");
  button(root, "Delete").click();
  expect(onDelete).toHaveBeenCalledWith("a1");
});

it("explains how to add labels when empty and disables export", () => {
  const { root } = mount([]);
  expect(root.textContent).toContain("Click a peak on the spectra to add a label.");
  expect(button(root, "Export labels…").disabled).toBe(true);
});

it("offers import and export", () => {
  const { root, onImport, onExport } = mount([{ id: "a1", ppm: 1, label: "x", angle: 0 }]);
  button(root, "Import labels…").click();
  button(root, "Export labels…").click();
  expect(onImport).toHaveBeenCalledOnce();
  expect(onExport).toHaveBeenCalledOnce();
});

it("picks a label to edit", () => {
  const { root, onEdit } = mount([{ id: "a1", ppm: 1, label: "lac", angle: 0 }]);
  button(root, "lac").click();
  expect(onEdit).toHaveBeenCalledWith("a1");
});
