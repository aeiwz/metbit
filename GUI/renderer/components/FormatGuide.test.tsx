import { render } from "preact";
import { expect, it, vi } from "vitest";
import { FormatGuide } from "./FormatGuide";

function mount(source: string) {
  const root = document.createElement("div");
  const onSaveExample = vi.fn();
  render(<FormatGuide source={source} onSaveExample={onSaveExample} />, root);
  return { root, onSaveExample };
}

it("shows the processed table mock-up and checklist for CSV", () => {
  const { root, onSaveExample } = mount("csv");
  expect(root.textContent).toContain("Processed data table");
  const headers = [...root.querySelectorAll(".mock-table thead tr:last-child th")].map((th) => th.textContent);
  expect(headers.slice(0, 4)).toEqual(["Sample", "Group", "Age", "10.000"]);
  expect(root.querySelectorAll(".steps li").length).toBe(6);
  [...root.querySelectorAll("button")].find((b) => b.textContent === "Save example CSV")!.click();
  expect(onSaveExample).toHaveBeenCalledOnce();
});

it("shows the Bruker folder, metadata link and checklist for Bruker", () => {
  const { root } = mount("bruker");
  expect(root.textContent).toContain("Raw Bruker + metadata");
  expect(root.textContent).toContain("pdata/1/procs");
  const linked = [...root.querySelectorAll(".link")].map((e) => e.textContent);
  expect(linked.filter((t) => t?.startsWith("61020601"))).toHaveLength(2); // folder and metadata cell
  expect(root.querySelectorAll(".steps li").length).toBe(5);
  expect(root.textContent).not.toContain("Save example CSV");
});
