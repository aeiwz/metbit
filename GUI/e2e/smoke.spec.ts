import { _electron as electron, expect, test, type Page } from "@playwright/test";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

function makeCsv(groups = ["A", "B"], perGroup = 6): string {
  const ppm = Array.from({ length: 40 }, (_, i) => (10 - i * 0.25).toFixed(4));
  const n = groups.length * perGroup;
  const rows = Array.from({ length: n }, (_, i) => {
    const g = Math.floor(i / perGroup);
    const values = ppm.map((_, j) => (Math.abs(Math.sin(i * 7 + j)) + 1 + (j === 5 ? g * 1.5 : 0)).toFixed(5));
    return [`s${i}`, groups[g], (20 + i * 3).toFixed(1), ...values].join(",");
  });
  const file = join(mkdtempSync(join(tmpdir(), "metbit-e2e-")), "data.csv");
  writeFileSync(file, [["Sample", "Group", "Age", ...ppm].join(","), ...rows].join("\n"));
  return file;
}

const run = (win: Page) => win.getByRole("button", { name: "Run", exact: true }).click();
const stageButton = (win: Page, name: string) => win.locator(".rail button", { hasText: name });
const plot = (win: Page) => expect(win.locator(".plot .main-svg").first()).toBeVisible({ timeout: 30_000 });

async function launchWithImport(csv: string) {
  const app = await electron.launch({ args: ["."] });
  const win = await app.firstWindow();
  await win.fill("#f-path", csv);
  await run(win);
  await expect(win.locator("table.data")).toContainText("s11");
  return { app, win };
}

test("import a CSV and run PCA", async () => {
  const { app, win } = await launchWithImport(makeCsv());
  await stageButton(win, "PCA").click();
  await run(win);
  await plot(win);
  await app.close();
});

test("normalize, OPLS-DA, STOCSY and volcano", async () => {
  const { app, win } = await launchWithImport(makeCsv());

  await stageButton(win, "Normalize").click();
  await run(win);
  await plot(win);

  await stageButton(win, "OPLS-DA").click();
  await expect(win.locator("#f-group_a")).toHaveValue("A");
  await expect(win.locator("#f-group_b")).toHaveValue("B");
  await run(win);
  await plot(win);
  await win.getByRole("tab", { name: "VIP" }).click();
  await plot(win);

  await stageButton(win, "STOCSY").click();
  await win.fill("#f-anchor_ppm", "5");
  await run(win);
  await plot(win);

  await stageButton(win, "Volcano").click();
  await run(win);
  await expect(win.getByRole("tab", { name: "Volcano" })).toBeVisible();
  await plot(win);

  await app.close();
});

test("PLS-R, LDA, ICA, HCA and group tests", async () => {
  const { app, win } = await launchWithImport(makeCsv(["A", "B", "C"], 4));

  await stageButton(win, "PLS-R").click();
  await expect(win.locator("#f-target")).toHaveValue("Age");
  await run(win);
  await expect(win.getByRole("tab", { name: "Predicted vs actual" })).toBeVisible();
  await plot(win);

  for (const name of ["LDA", "ICA"]) {
    await stageButton(win, name).click();
    await run(win);
    await plot(win);
  }

  await stageButton(win, "HCA").click();
  await run(win);
  await plot(win);
  await win.getByRole("tab", { name: "Heatmap" }).click();
  await plot(win);

  await stageButton(win, "Group tests").click();
  await run(win);
  await plot(win);
  await win.getByRole("tab", { name: "Table" }).click();
  await expect(win.locator("table.data")).toContainText("p_adj");

  await app.close();
});
