import { render } from "preact";
import { expect, it } from "vitest";
import { MoleculeLoader } from "./MoleculeLoader";

function mount(progress?: number | null) {
  const root = document.createElement("div");
  render(<MoleculeLoader caption="Running OPLS-DA…" progress={progress} />, root);
  return root;
}

it("draws a benzene ring with alternating double bonds", () => {
  const root = mount();
  expect(root.querySelector(".loader")!.getAttribute("role")).toBe("status");
  expect(root.querySelectorAll(".molecule .atom")).toHaveLength(6);
  expect(root.querySelectorAll(".molecule .bond")).toHaveLength(6);
  expect(root.querySelectorAll(".molecule .double")).toHaveLength(6);
  expect(root.querySelectorAll(".molecule .double.alt")).toHaveLength(3);
  expect(root.querySelector(".loader-caption")!.textContent).toBe("Running OPLS-DA…");
});

it("shows a percentage only when there is progress", () => {
  expect(mount(0.42).querySelector(".loader-progress")!.textContent).toBe("42%");
  expect(mount(null).querySelector(".loader-progress")).toBeNull();
  expect(mount().querySelector(".loader-progress")).toBeNull();
});
