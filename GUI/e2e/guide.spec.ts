import { _electron as electron, expect, test } from "@playwright/test";
import { existsSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

test("format guide follows Source and its example CSV imports", async () => {
  const dir = mkdtempSync(join(tmpdir(), "metbit-e2e-"));
  const env = { ...process.env, METBIT_E2E_DIR: dir, METBIT_USER_DATA: join(dir, "user") } as Record<string, string>;
  const app = await electron.launch({ args: ["."], env });
  const win = await app.firstWindow();
  const guide = win.locator(".guide");

  await expect(guide).toContainText("Processed data table");
  await win.selectOption("#f-source", "bruker");
  await expect(guide).toContainText("pdata/1/procs");
  await win.selectOption("#f-source", "csv");
  await expect(guide).toContainText("Processed data table");

  await win.getByRole("button", { name: "Save example CSV" }).click();
  const example = join(dir, "metbit-example.csv");
  await expect.poll(() => existsSync(example)).toBe(true);
  await expect(win.locator("#f-path")).toHaveValue(example);

  await win.getByRole("button", { name: "Run", exact: true }).click();
  await expect(win.locator("table.data")).toContainText("s12");
  await expect(guide).toHaveCount(0);
  await win.getByRole("button", { name: "File format" }).click();
  await expect(guide).toBeVisible();

  await app.close();
});
