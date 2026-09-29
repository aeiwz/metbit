import { _electron as electron, expect, test, type Page } from "@playwright/test";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const TIMES = ["1-wk", "2-wk", "10-wk"];

// 2 groups x 4 subjects x 3 time points; a peak at columns 5-9 rises over time in group B only
function makeCsv(dir: string): string {
  const ppm = Array.from({ length: 40 }, (_, j) => (10 - j * 0.25).toFixed(4));
  const rows: string[] = [];
  for (const g of ["A", "B"])
    for (let s = 0; s < 4; s++)
      TIMES.forEach((time, t) => {
        const values = ppm.map((_, j) =>
          (Math.abs(Math.sin(s * 5 + t * 3 + j)) * 0.2 + 1 + s * 0.05 + (g === "B" && j >= 5 && j < 10 ? 3 * t : 0)).toFixed(5),
        );
        rows.push([`${g}${s}-${time}`, g, time, `${g}${s}`, ...values].join(","));
      });
  const file = join(dir, "ts.csv");
  writeFileSync(file, [["Sample", "Group", "Time", "Subject", ...ppm].join(","), ...rows].join("\n"));
  return file;
}

const mainTitle = (win: Page) =>
  win.evaluate(() => (document.querySelector(".split-main .plot") as any)?.layout?.title?.text ?? "");

async function clickCompanion(win: Page, fx: number) {
  const box = (await win.locator(".split-companion .nsewdrag").first().boundingBox())!;
  await win.mouse.move(box.x + box.width * fx, box.y + box.height * 0.5);
  await win.mouse.click(box.x + box.width * fx, box.y + box.height * 0.5);
}

test("time series: trajectory, time profile, ASCA and mixed model", async () => {
  test.setTimeout(120_000);
  const dir = mkdtempSync(join(tmpdir(), "metbit-e2e-"));
  const app = await electron.launch({ args: ["."], env: { ...process.env, METBIT_USER_DATA: join(dir, "user") } as Record<string, string> });
  const win = await app.firstWindow();
  const run = () => win.getByRole("button", { name: "Run", exact: true }).click();
  const open = (name: string) => win.locator(".rail button", { hasText: name }).first().click();

  await win.fill("#f-path", makeCsv(dir));
  await run();
  await expect(win.locator("table.data")).toContainText("B3-10-wk");

  // the pickers default to the time-like and subject-like columns
  await open("Trajectory");
  await expect(win.locator("#f-time_column")).toHaveValue("Time");
  await expect(win.locator("#f-subject_column")).toHaveValue("Subject");
  await win.locator("#f-multilevel").check();
  await run();
  await expect.poll(() => win.evaluate(() => (document.querySelector(".plot") as any)?.layout?.title?.text ?? "")).toContain("multilevel");
  await win.getByRole("tab", { name: "Subjects" }).click();
  await expect(win.locator(".plot .main-svg").first()).toBeVisible();

  // time profile follows a click on the spectrum below it
  await open("Time profile");
  await run();
  await expect(win.locator(".split-companion .plot .main-svg").first()).toBeVisible({ timeout: 30_000 });
  const before = await mainTitle(win);
  expect(before).toContain("ppm over time");
  await clickCompanion(win, 0.9);
  await expect.poll(() => mainTitle(win)).not.toBe(before);

  // ASCA effect table
  await open("ASCA");
  await win.fill("#f-permutations", "30");
  await win.fill("#f-bin_width", "0");
  await run();
  await win.getByRole("tab", { name: "Table" }).click();
  await expect(win.locator("table.data")).toContainText("Time × Group");

  // mixed model: ranked table, and a click on the p-value plot picks the profile
  await open("Mixed model");
  await win.fill("#f-bin_width", "0");
  await run();
  await win.getByRole("tab", { name: "Table" }).click();
  await expect(win.locator("table.data thead")).toContainText("p_interaction_adj");
  await win.getByRole("tab", { name: "Profile" }).click();
  await expect(win.locator(".split-companion .plot .main-svg").first()).toBeVisible();
  const top = await mainTitle(win);
  await clickCompanion(win, 0.05);
  await expect.poll(() => mainTitle(win)).not.toBe(top);

  await app.close();
});
