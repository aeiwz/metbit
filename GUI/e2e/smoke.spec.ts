import { _electron as electron, expect, test } from "@playwright/test";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

function makeCsv(): string {
  const ppm = Array.from({ length: 40 }, (_, i) => (10 - i * 0.25).toFixed(4));
  const rows = Array.from({ length: 12 }, (_, i) => {
    const group = i < 6 ? "A" : "B";
    const values = ppm.map((_, j) => (Math.sin(i * 7 + j) + (group === "B" && j === 5 ? 3 : 0)).toFixed(5));
    return [`s${i}`, group, ...values].join(",");
  });
  const file = join(mkdtempSync(join(tmpdir(), "metbit-e2e-")), "data.csv");
  writeFileSync(file, [["Sample", "Group", ...ppm].join(","), ...rows].join("\n"));
  return file;
}

test("import a CSV and run PCA", async () => {
  const app = await electron.launch({ args: ["."] });
  const win = await app.firstWindow();

  await win.fill("#f-path", makeCsv());
  await win.getByRole("button", { name: "Run" }).click();
  await expect(win.locator("table.data")).toContainText("s11");

  await win.getByRole("button", { name: "PCA" }).click();
  await win.getByRole("button", { name: "Run" }).click();
  await expect(win.locator(".plot .main-svg").first()).toBeVisible({ timeout: 30_000 });

  await app.close();
});
