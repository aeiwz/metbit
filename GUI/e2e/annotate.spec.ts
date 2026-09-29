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

const plotLabels = (win: Page) =>
  win.evaluate(() => ((document.querySelector(".plot") as any)?.layout?.annotations ?? []).map((a: any) => a.text));

test("label a peak, see it on other spectra and in the intensity table", async () => {
  const dir = mkdtempSync(join(tmpdir(), "metbit-e2e-"));
  const env = { ...process.env, METBIT_USER_DATA: join(dir, "user") } as Record<string, string>;
  const app = await electron.launch({ args: ["."], env });
  const win = await app.firstWindow();
  const run = () => win.getByRole("button", { name: "Run", exact: true }).click();
  const plot = () => expect(win.locator(".plot .main-svg").first()).toBeVisible({ timeout: 30_000 });

  await win.fill("#f-path", makeCsv(dir));
  await run();
  await expect(win.locator("table.data")).toContainText("s11");

  await win.locator(".rail button", { hasText: "Annotate" }).click();
  await run();
  await plot();
  const box = (await win.locator(".plot .nsewdrag").first().boundingBox())!;
  await win.mouse.move(box.x + box.width * 0.2, box.y + box.height * 0.3);
  await win.mouse.click(box.x + box.width * 0.2, box.y + box.height * 0.3);
  await win.getByLabel("Label", { exact: true }).fill("lactate");
  await win.getByLabel("Label", { exact: true }).press("Enter");

  await expect(win.locator(".label-list")).toContainText("lactate");
  await expect.poll(() => plotLabels(win)).toContain("lactate");
  await win.getByRole("tab", { name: "Table" }).click();
  await expect(win.locator("table.data thead")).toContainText("lactate_");

  await win.locator(".rail button", { hasText: "Normalize" }).click();
  await run();
  await plot();
  await expect.poll(() => plotLabels(win)).toContain("lactate");

  await app.close();
});
