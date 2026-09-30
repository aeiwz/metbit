import { _electron as electron, expect, test, type Page } from "@playwright/test";
import { execSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
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

const run = (win: Page) => win.getByRole("button", { name: "Run", exact: true }).click();
const done = (win: Page, name: string) => expect(win.locator(".rail button", { hasText: name }).locator(".dot.done")).toBeVisible();
const plot = (win: Page) => expect(win.locator(".plot .main-svg").first()).toBeVisible({ timeout: 30_000 });

async function launch(env: Record<string, string>) {
  const app = await electron.launch({ args: ["."], env: { ...process.env, ...env } as Record<string, string> });
  return { app, win: await app.firstWindow() };
}

async function importAndPca(win: Page, csv: string) {
  await win.getByRole("button", { name: "New analysis" }).click(); // past the landing page
  await win.fill("#f-path", csv);
  await run(win);
  await expect(win.locator("table.data")).toContainText("s11");
  await win.locator(".rail button", { hasText: "PCA" }).click();
  await run(win);
  await plot(win);
}

test("export CSV and PNG, save a project and reopen it", async () => {
  const dir = mkdtempSync(join(tmpdir(), "metbit-e2e-"));
  const { app, win } = await launch({ METBIT_E2E_DIR: dir, METBIT_USER_DATA: join(dir, "user") });
  await importAndPca(win, makeCsv(dir));

  await win.getByRole("button", { name: "Export CSV" }).click();
  await expect.poll(() => existsSync(join(dir, "pca.csv"))).toBe(true);
  await win.getByRole("button", { name: "PNG", exact: true }).click();
  await expect.poll(() => existsSync(join(dir, "pca-scores.png"))).toBe(true);
  expect(readFileSync(join(dir, "pca-scores.png")).subarray(1, 4).toString()).toBe("PNG");
  await win.getByRole("button", { name: "Save project…" }).click();
  await expect.poll(() => existsSync(join(dir, "analysis.metbit"))).toBe(true);
  await app.close();

  const again = await launch({ METBIT_E2E_OPEN: join(dir, "analysis.metbit"), METBIT_USER_DATA: join(dir, "user2") });
  await again.win.getByRole("button", { name: "Open project…" }).click();
  await done(again.win, "PCA");
  await again.win.locator(".rail button", { hasText: "PCA" }).click();
  await plot(again.win);
  await again.app.close();
});

test("a backend crash restores the session after Restart", async () => {
  const dir = mkdtempSync(join(tmpdir(), "metbit-e2e-"));
  const { app, win } = await launch({ METBIT_USER_DATA: join(dir, "user") });
  await importAndPca(win, makeCsv(dir));

  // only this app's sidecar (a direct child of its main process): other tests run in parallel workers
  execSync(`pkill -P ${app.process().pid} -f metbit_server`);
  await expect(win.getByRole("alert")).toContainText("Backend stopped");
  await win.getByRole("button", { name: "Restart" }).click();
  await expect(win.getByText("Session restored")).toBeVisible({ timeout: 30_000 });
  await done(win, "PCA");
  await app.close();
});
