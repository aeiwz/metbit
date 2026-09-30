import { render } from "preact";
import { expect, it, vi } from "vitest";
import type { Annotation } from "../api";
import { AnnotatePanels } from "./AnnotatePanels";

const tick = () => new Promise((r) => setTimeout(r));

function mount(peak: number | null, lineStyles = {}, editing: Annotation | null = null) {
  const root = document.createElement("div");
  const handlers = { onAdd: vi.fn(), onApplyStyle: vi.fn(), onResetStyle: vi.fn(), onSave: vi.fn(), onCancelEdit: vi.fn() };
  render(
    <AnnotatePanels
      peak={peak}
      targets={{ groups: ["A", "B"], samples: ["s0", "s1"] }}
      lineStyles={lineStyles}
      editing={editing}
      exports={<button>Download HTML</button>}
      {...handlers}
    >
      <ul class="label-list" />
    </AnnotatePanels>,
    root,
  );
  return { root, ...handlers };
}

const type = (el: HTMLInputElement, value: string) => {
  el.value = value;
  el.dispatchEvent(new Event("input"));
};
const button = (root: HTMLElement, t: string) => [...root.querySelectorAll("button")].find((b) => b.textContent === t) as HTMLButtonElement;

it("adds a label at the current peak", async () => {
  const { root, onAdd } = mount(3.03);
  expect(button(root, "Add Label").disabled).toBe(true);
  type(root.querySelector<HTMLInputElement>("#annotate-label")!, "lactate");
  await tick();
  expect(button(root, "Add Label").disabled).toBe(false);
  root.querySelector("form")!.dispatchEvent(new Event("submit", { cancelable: true }));
  expect(onAdd).toHaveBeenCalledWith("lactate", 0);
});

it("needs a clicked peak before adding", async () => {
  const { root } = mount(null);
  type(root.querySelector<HTMLInputElement>("#annotate-label")!, "lactate");
  await tick();
  expect(button(root, "Add Label").disabled).toBe(true);
  expect(root.textContent).toContain("Click a peak on the spectra, or a label to edit it.");
});

it("styles a group or a sample line and can reset it", async () => {
  const { root, onApplyStyle, onResetStyle } = mount(null, { B: { dash: "dot" } });
  const pick = root.querySelector<HTMLSelectElement>('[aria-label="Line to style"]')!;
  expect([...pick.querySelectorAll("optgroup")].map((g) => g.label)).toEqual(["Groups", "Samples"]);
  pick.value = "B";
  pick.dispatchEvent(new Event("change"));
  const dash = root.querySelector<HTMLSelectElement>('[aria-label="Line dash"]')!;
  dash.value = "dot";
  dash.dispatchEvent(new Event("change"));
  await tick();
  button(root, "Apply Style").click();
  expect(onApplyStyle).toHaveBeenCalledWith("B", expect.objectContaining({ dash: "dot" }));
  button(root, "Reset").click();
  expect(onResetStyle).toHaveBeenCalledWith("B");
  expect(root.textContent).toContain("Download HTML");
});

it("edits a picked label's text and angle", async () => {
  const { root, onSave, onCancelEdit, onAdd } = mount(null, {}, { id: "a1", ppm: 2.5, label: "&lt;lac", angle: 30 });
  expect(root.textContent).toContain("Edit annotation");
  const label = root.querySelector<HTMLInputElement>("#annotate-label")!;
  await tick();
  expect(label.value).toBe("<lac"); // decoded for editing
  expect(root.querySelector<HTMLInputElement>('[aria-label="Angle"]')!.value).toBe("30");
  type(label, "lactate");
  await tick();
  root.querySelector("form")!.dispatchEvent(new Event("submit", { cancelable: true }));
  expect(onSave).toHaveBeenCalledWith("a1", "lactate", 30);
  expect(onAdd).not.toHaveBeenCalled();
  button(root, "Cancel").click();
  expect(onCancelEdit).toHaveBeenCalled();
});
