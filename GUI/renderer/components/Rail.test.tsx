import { render } from "preact";
import { expect, it } from "vitest";
import type { StageInfo } from "../api";
import { Rail } from "./Rail";

const stage = (id: string, section: string, requires: string | null = null, status: StageInfo["status"] = "empty"): StageInfo => ({
  id,
  title: id.toUpperCase(),
  section,
  requires,
  status,
  views: [],
  ppm_views: {},
  has_table: true,
  schema: { properties: {} },
  params: null,
});

it("renders one header per section and gates stages on their requirement", () => {
  const root = document.createElement("div");
  const stages = [stage("import", "Data"), stage("normalize", "Data", "import"), stage("pca", "Analysis", "import")];
  render(<Rail stages={stages} current="import" onSelect={() => {}} />, root);
  expect([...root.querySelectorAll(".rail-section")].map((e) => e.textContent)).toEqual(["Data", "Analysis"]);
  const button = (text: string) => [...root.querySelectorAll("button")].find((b) => b.textContent === text)!;
  expect(button("IMPORT").disabled).toBe(false);
  expect(button("PCA").disabled).toBe(true);
});
