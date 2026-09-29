import { _electron as electron, expect, test } from "@playwright/test";
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const HEART = join(__dirname, "..", "..", "Benchmark", "data", "MTBLS2052", "proc_subset_heart_wk6");
const MAP = join(HEART, "..", "noesy_sample_map.csv");

test.skip(!existsSync(HEART), "MTBLS2052 subset is local-only benchmark data");

test("Bruker folder to processed spectra to PCA", async () => {
  // the sample map covers every matrix; the join only needs the rows for these experiments
  const meta = join(mkdtempSync(join(tmpdir(), "metbit-e2e-")), "heart_meta.csv");
  writeFileSync(meta, readFileSync(MAP, "utf8"));

  const app = await electron.launch({ args: ["."] });
  const win = await app.firstWindow();
  const run = () => win.getByRole("button", { name: "Run", exact: true }).click();

  await win.selectOption("#f-source", "bruker");
  await win.fill("#f-folder", HEART);
  await win.fill("#f-metadata_path", meta);
  await win.fill("#f-sample_column", "fid");
  await win.fill("#f-group_column", "Factor Value[nephrectomy]");
  await run();
  await expect(win.locator("table.data")).toContainText("61020601");

  await win.locator(".rail button", { hasText: "Process" }).click();
  await run();
  // per-sample progress drives the molecule loader's percentage while experiments process
  await expect(win.locator(".loader .loader-caption")).toHaveText("Running Process…");
  await expect(win.locator(".loader-progress")).toBeVisible({ timeout: 10_000 });
  await expect(win.locator(".plot .main-svg").first()).toBeVisible({ timeout: 60_000 });

  await win.locator(".rail button", { hasText: "PCA" }).click();
  await run();
  await expect(win.locator(".plot .main-svg").first()).toBeVisible({ timeout: 30_000 });

  await app.close();
});
