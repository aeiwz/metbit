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

const importSchema: JsonSchema = {
  properties: {
    source: { type: "string", title: "Source", enum: ["csv", "bruker"], default: "csv" },
    path: { type: "string", title: "CSV file", format: "file-path", show_if: { source: "csv" } },
    folder: { type: "string", title: "Bruker folder", format: "dir-path", show_if: { source: "bruker" } },
    auto_phasing: { type: "boolean", title: "Automatic phasing", default: false },
  },
};

it("shows only the fields whose show_if matches", async () => {
  const { root } = mount({ schema: importSchema });
  expect(root.querySelector("#f-path")).not.toBeNull();
  expect(root.querySelector("#f-folder")).toBeNull();
  const source = root.querySelector<HTMLSelectElement>("#f-source")!;
  source.value = "bruker";
  source.dispatchEvent(new Event("change"));
  await tick();
  expect(root.querySelector("#f-path")).toBeNull();
  expect(root.querySelector("#f-folder")).not.toBeNull();
});

it("renders booleans as checkboxes and submits booleans", async () => {
  const { root, onRun } = mount({ schema: importSchema });
  const box = root.querySelector<HTMLInputElement>("#f-auto_phasing")!;
  expect(box.type).toBe("checkbox");
  box.click();
  await tick();
  root.querySelector("form")!.dispatchEvent(new Event("submit", { cancelable: true }));
  expect(onRun.mock.calls[0][0].auto_phasing).toBe(true);
});

it("opens a folder picker for dir-path fields", async () => {
  const openFile = vi.fn().mockResolvedValue("/data/exps");
  (window as any).metbit = { openFile };
  const { root } = mount({ schema: importSchema, initial: { source: "bruker" } });
  [...root.querySelectorAll("button")].find((b) => b.textContent === "Browse")!.click();
  await tick();
  expect(openFile).toHaveBeenCalledWith(undefined, true);
  await tick();
  expect(root.querySelector<HTMLInputElement>("#f-folder")!.value).toBe("/data/exps");
});

it("reports live values to onValuesChange", async () => {
  const seen: unknown[] = [];
  const { root } = mount({ schema: importSchema, onValuesChange: (v) => seen.push(v.source) });
  await tick();
  const source = root.querySelector<HTMLSelectElement>("#f-source")!;
  source.value = "bruker";
  source.dispatchEvent(new Event("change"));
  // useEffect runs after paint, so wait rather than assume one tick is enough
  await vi.waitFor(() => expect(seen[seen.length - 1]).toBe("bruker"));
  expect(seen[0]).toBe("csv");
});

it("merges injected values without losing other edits", async () => {
  const stocsy: JsonSchema = {
    properties: {
      anchor_ppm: { type: "number", title: "Anchor (ppm)", pick_ppm: true },
      p_threshold: { type: "number", title: "p", default: 0.0001 },
    },
  };
  const root = document.createElement("div");
  const onRun = vi.fn();
  const view = (injected: { values: Record<string, unknown>; n: number } | null) =>
    render(<ParamPanel title="STOCSY" schema={stocsy} initial={null} errors={{}} busy={false} onRun={onRun} injected={injected} />, root);
  view(null);
  const p = root.querySelector<HTMLInputElement>("#f-p_threshold")!;
  p.value = "0.01";
  p.dispatchEvent(new Event("input"));
  await tick();
  view({ values: { anchor_ppm: 3.03 }, n: 1 });
  await vi.waitFor(() => expect(root.querySelector<HTMLInputElement>("#f-anchor_ppm")!.value).toBe("3.03"));
  root.querySelector("form")!.dispatchEvent(new Event("submit", { cancelable: true }));
  expect(onRun).toHaveBeenCalledWith({ anchor_ppm: 3.03, p_threshold: 0.01 });
});

it("accepts scientific notation in decimal fields and keeps unfinished input as typed", async () => {
  const pSchema: JsonSchema = { properties: { p_threshold: { type: "number", title: "p threshold", default: 0.0001 } } };
  const { root, onRun } = mount({ schema: pSchema });
  const p = root.querySelector<HTMLInputElement>("#f-p_threshold")!;
  const type = async (text: string) => {
    p.value = text;
    p.dispatchEvent(new Event("input"));
    await tick();
  };
  await type("1e-");
  expect(p.value).toBe("1e-"); // not wiped while typing
  await type("1e-7");
  root.querySelector("form")!.dispatchEvent(new Event("submit", { cancelable: true }));
  expect(onRun).toHaveBeenLastCalledWith({ p_threshold: 1e-7 });
  await type("abc");
  root.querySelector("form")!.dispatchEvent(new Event("submit", { cancelable: true }));
  expect(onRun).toHaveBeenLastCalledWith({ p_threshold: "abc" }); // the server's 422 explains it on the field
});

it("explains number errors in plain words", () => {
  expect(fieldErrors([{ loc: ["p_threshold"], msg: "Input should be a valid number, unable to parse string as a number" }])).toEqual({
    p_threshold: "Enter a number, for example 0.0001 or 1e-7",
  });
});
