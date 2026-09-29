import { render } from "preact";
import { expect, it, vi } from "vitest";
import type { JsonSchema } from "../api";
import { ParamPanel, fieldErrors } from "./ParamPanel";

const schema: JsonSchema = {
  properties: {
    path: { type: "string", title: "CSV file", format: "file-path" },
    n_components: { type: "integer", title: "Components", default: 2, minimum: 2, maximum: 10 },
    scaling: { type: "string", title: "Scaling", enum: ["pareto", "uv", "mean"], default: "pareto" },
  },
  required: ["path"],
};

function mount(props: Partial<Parameters<typeof ParamPanel>[0]> = {}) {
  const root = document.createElement("div");
  const onRun = vi.fn();
  render(
    <ParamPanel title="PCA" schema={schema} initial={null} errors={{}} busy={false} onRun={onRun} {...props} />,
    root,
  );
  return { root, onRun };
}

it("renders one control per property with the right kind", () => {
  const { root } = mount();
  expect(root.querySelector<HTMLInputElement>("#f-n_components")!.type).toBe("number");
  expect(root.querySelector("#f-scaling")!.tagName).toBe("SELECT");
  expect(root.querySelector("#f-path")).not.toBeNull();
  expect(root.textContent).toContain("Browse");
  expect(root.textContent).toContain("Components");
});

it("submits defaults merged with initial values", () => {
  const { root, onRun } = mount({ initial: { path: "/tmp/a.csv" } });
  root.querySelector("form")!.dispatchEvent(new Event("submit", { cancelable: true }));
  expect(onRun).toHaveBeenCalledWith({ path: "/tmp/a.csv", n_components: 2, scaling: "pareto" });
});

it("marks fields named in errors as invalid and shows form-level errors", () => {
  const { root } = mount({ errors: { n_components: "too small", "": "axes clash" } });
  expect(root.querySelector("#f-n_components")!.closest(".field")!.classList.contains("invalid")).toBe(true);
  expect(root.textContent).toContain("too small");
  expect(root.textContent).toContain("axes clash");
});

it("disables Run while busy", () => {
  const { root } = mount({ busy: true });
  expect(root.querySelector<HTMLButtonElement>("button[type=submit]")!.disabled).toBe(true);
});

it("maps pydantic 422 detail to field errors", () => {
  expect(fieldErrors([{ loc: ["n_components"], msg: "bad" }, { loc: [], msg: "whole" }])).toEqual({
    n_components: "bad",
    "": "whole",
  });
  expect(fieldErrors("boom")).toEqual({ "": "boom" });
});
