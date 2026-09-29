import { _electron as electron, expect, test, type Page } from "@playwright/test";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

function makeCsv(): string {
  const ppm = Array.from({ length: 40 }, (_, i) => (10 - i * 0.25).toFixed(4));
  const rows = Array.from({ length: 12 }, (_, i) => {
    const group = i < 6 ? "A" : "B";
    const values = ppm.map((_, j) => (Math.abs(Math.sin(i * 7 + j)) + 1 + (group === "B" && j === 5 ? 3 : 0)).toFixed(5));
    return [`s${i}`, group, ...values].join(",");
  });
  const file = join(mkdtempSync(join(tmpdir(), "metbit-e2e-")), "data.csv");
  writeFileSync(file, [["Sample", "Group", ...ppm].join(","), ...rows].join("\n"));
  return file;
}

const run = (win: Page) => win.getByRole("button", { name: "Run", exact: true }).click();
const stageButton = (win: Page, name: string) => win.locator(".rail button", { hasText: name });
const plot = (win: Page) => expect(win.locator(".plot .main-svg").first()).toBeVisible({ timeout: 30_000 });

test("import a CSV and run PCA", async () => {
  const app = await electron.launch({ args: ["."] });
  const win = await app.firstWindow();

  await win.fill("#f-path", makeCsv());
  await run(win);
  await expect(win.locator("table.data")).toContainText("s11");

  await stageButton(win, "PCA").click();
  await run(win);
  await plot(win);

  await app.close();
});

test("normalize, OPLS-DA, STOCSY and volcano", async () => {
  const app = await electron.launch({ args: ["."] });
  const win = await app.firstWindow();

  await win.fill("#f-path", makeCsv());
  await run(win);
  await expect(stageButton(win, "Normalize")).toBeEnabled();

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
