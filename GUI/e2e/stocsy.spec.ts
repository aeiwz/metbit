import { _electron as electron, expect, test } from "@playwright/test";
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

test("STOCSY: click the spectra to set the anchor, run, reuse and clear", async () => {
  const dir = mkdtempSync(join(tmpdir(), "metbit-e2e-"));
  const env = { ...process.env, METBIT_USER_DATA: join(dir, "user") } as Record<string, string>;
  const app = await electron.launch({ args: ["."], env });
  const win = await app.firstWindow();
  const run = () => win.getByRole("button", { name: "Run", exact: true }).click();
  const plot = () => expect(win.locator(".plot .main-svg").first()).toBeVisible({ timeout: 30_000 });

  await win.fill("#f-path", makeCsv(dir));
  await run();
  await expect(win.locator("table.data")).toContainText("s11");

  await win.locator(".rail button", { hasText: "STOCSY" }).click();
  await expect(win.getByRole("tab", { name: "Spectra" })).toBeVisible();
  await plot(); // spectra before any run

  const box = (await win.locator(".plot .nsewdrag").first().boundingBox())!;
  await win.mouse.move(box.x + box.width * 0.3, box.y + box.height * 0.3);
  await win.mouse.click(box.x + box.width * 0.3, box.y + box.height * 0.3);
  await expect.poll(() => win.inputValue("#f-anchor_ppm")).not.toBe("");

  await run();
  await expect(win.getByRole("tab", { name: "Correlation" })).toHaveAttribute("aria-selected", "true");
  await plot();
  await expect(win.locator(".stocsy-tools .label-list")).toContainText("ppm");

  // a rejected run must not be recorded as a recent anchor
  await win.fill("#f-p_threshold", "abc");
  await run();
  await expect(win.locator(".field.invalid")).toContainText("Enter a number");
  await expect(win.locator(".stocsy-tools .label-list li")).toHaveCount(1);
  // scientific notation types and runs
  await win.fill("#f-p_threshold", "1e-7");
  await run();
  await expect(win.locator(".stocsy-tools .label-list")).toContainText("p 1e-7");

  await win.getByRole("button", { name: "Clear anchor" }).click();
  await expect(win.locator("#f-anchor_ppm")).toHaveValue("");

  await app.close();
});
