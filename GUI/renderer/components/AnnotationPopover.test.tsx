import { render } from "preact";
import { expect, it, vi } from "vitest";
import { AnnotationPopover } from "./AnnotationPopover";

const tick = () => new Promise((r) => setTimeout(r));

function mount() {
  const root = document.createElement("div");
  const onAdd = vi.fn();
  const onCancel = vi.fn();
  render(<AnnotationPopover ppm={1.3312} x={100} y={200} onAdd={onAdd} onCancel={onCancel} />, root);
  return { root, onAdd, onCancel };
}

const type = (el: HTMLInputElement, value: string) => {
  el.value = value;
  el.dispatchEvent(new Event("input"));
};

it("adds the typed label and angle", async () => {
  const { root, onAdd } = mount();
  expect(root.textContent).toContain("1.331 ppm");
  type(root.querySelector<HTMLInputElement>('[aria-label="Label"]')!, "  lactate ");
  type(root.querySelector<HTMLInputElement>('[aria-label="Angle"]')!, "15");
  await tick();
  root.querySelector("form")!.dispatchEvent(new Event("submit", { cancelable: true }));
  expect(onAdd).toHaveBeenCalledWith("lactate", 15);
});

it("disables Add while the label is blank", () => {
  const { root } = mount();
  expect([...root.querySelectorAll("button")].find((b) => b.textContent === "Add")!.disabled).toBe(true);
});

it("cancels on Escape", () => {
  const { root, onCancel } = mount();
  root.querySelector("form")!.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
  expect(onCancel).toHaveBeenCalledOnce();
});
