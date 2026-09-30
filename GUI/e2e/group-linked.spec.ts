import { _electron as electron, expect, test, type Page } from "@playwright/test";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

function makeCsv(dir: string): string {
  const ppm = Array.from({ length: 40 }, (_, i) => (10 - i * 0.25).toFixed(4));
  const rows = Array.from({ length: 12 }, (_, i) => {
    const group = i < 6 ? "A" : "B";
    const values = ppm.map((_, j) => (Math.abs(Math.sin(i * 7 + j)) + 1 + (group === "B" && j === 5 ? 3 : 0)).toFixed(5));
    return [`s${i}`, group, ...values].join(",");
  });
  const file = join(dir, "data.csv");
  writeFileSync(file, [["Sample", "Group", ...ppm].join(","), ...rows].join("\n"));
  return file;
}

const boxTitle = (win: Page) =>
  win.evaluate(() => (document.querySelector(".split-main .plot") as any)?.layout?.title?.text ?? "");

test("click the spectrum under the boxplot to test another ppm", async () => {
  const dir = mkdtempSync(join(tmpdir(), "metbit-e2e-"));
  const env = { ...process.env, METBIT_USER_DATA: join(dir, "user") } as Record<string, string>;
  const app = await electron.launch({ args: ["."], env });
  const win = await app.firstWindow();
  const run = () => win.getByRole("button", { name: "Run", exact: true }).click();

  await win.fill("#f-path", makeCsv(dir));
  await run();
  await expect(win.locator("table.data")).toContainText("s11");
  await win.locator(".rail button", { hasText: "Group tests" }).click();
  await win.fill("#f-bin_width", "0");
  await run();

  await expect(win.locator(".split-companion .plot .main-svg").first()).toBeVisible({ timeout: 30_000 });
  await expect(win.getByRole("tab", { name: "Spectra" })).toHaveCount(0); // companion is not a tab
  await expect.poll(() => boxTitle(win)).toContain("8.7500 ppm · ANOVA p = "); // top feature: the spiked peak
  // drag-zoom first: picking a peak must keep the zoom
  const range = () => win.evaluate(() => (document.querySelector(".split-companion .plot") as any)?.layout?.xaxis?.range);
  const box = (await win.locator(".split-companion .nsewdrag").first().boundingBox())!;
  await win.mouse.move(box.x + box.width * 0.05, box.y + box.height * 0.5);
  await win.mouse.down();
  await win.mouse.move(box.x + box.width * 0.9, box.y + box.height * 0.6, { steps: 6 });
  await win.mouse.up();
  await expect.poll(async () => (await range())?.[0]).toBeLessThan(9.9);
  const zoomed = await range();
  await win.waitForTimeout(500); // a click right after the zoom would count as a double-click (reset zoom)
  await win.mouse.move(box.x + box.width * 0.75, box.y + box.height * 0.5);
  await win.mouse.click(box.x + box.width * 0.75, box.y + box.height * 0.5);
  await expect.poll(() => boxTitle(win)).not.toContain("8.7500 ppm");
  await expect.poll(() => boxTitle(win)).toContain(" ppm · ANOVA p = ");
  expect(await range()).toEqual(zoomed); // zoom kept

  await app.close();
});
