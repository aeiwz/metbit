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

const tick = () => new Promise((r) => setTimeout(r));

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

it("shows field descriptions as hints", () => {
  const withHint: JsonSchema = { properties: { windows: { type: "string", title: "Windows", description: "e.g. 3.2-2.8" } } };
  const { root } = mount({ schema: withHint });
  expect(root.querySelector(".hint")!.textContent).toBe("e.g. 3.2-2.8");
});

it("offers the picked ppm for pick fields only", async () => {
  const pickSchema: JsonSchema = {
    properties: {
      anchor_ppm: { type: "number", title: "Anchor (ppm)", pick_ppm: true },
      p_threshold: { type: "number", title: "p", default: 0.0001 },
    },
  };
  const { root, onRun } = mount({ schema: pickSchema, picked: 3.0512 });
  const picks = root.querySelectorAll<HTMLButtonElement>(".pick");
  expect(picks).toHaveLength(1);
  expect(picks[0].textContent).toBe("Use 3.051 ppm");
  picks[0].click();
  await tick();
  root.querySelector("form")!.dispatchEvent(new Event("submit", { cancelable: true }));
  expect(onRun).toHaveBeenCalledWith({ anchor_ppm: 3.0512, p_threshold: 0.0001 });
});

it("hides the pick button when nothing is picked", () => {
  const pickSchema: JsonSchema = { properties: { anchor_ppm: { type: "number", pick_ppm: true } } };
  const { root } = mount({ schema: pickSchema, picked: null });
  expect(root.querySelector(".pick")).toBeNull();
});
