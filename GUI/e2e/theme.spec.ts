import { _electron as electron, expect, test, type Page } from "@playwright/test";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

function makeCsv(dir: string): string {
  const ppm = Array.from({ length: 20 }, (_, i) => (10 - i * 0.5).toFixed(4));
  const rows = Array.from({ length: 8 }, (_, i) => [`s${i}`, i < 4 ? "A" : "B", ...ppm.map((_, j) => (1 + ((i * 7 + j) % 5) / 10).toFixed(3))].join(","));
  const file = join(dir, "data.csv");
  writeFileSync(file, [["Sample", "Group", ...ppm].join(","), ...rows].join("\n"));
  return file;
}

const background = (win: Page) => win.evaluate(() => getComputedStyle(document.body).backgroundColor);
const plotText = (win: Page) => win.evaluate(() => (document.querySelector(".plot") as any)?.layout?.font?.color);

test("dark / light switch recolours the app and its plots, and is remembered", async () => {
  const dir = mkdtempSync(join(tmpdir(), "metbit-e2e-"));
  const env = { ...process.env, METBIT_USER_DATA: join(dir, "user") } as Record<string, string>;
  let app = await electron.launch({ args: ["."], env });
  let win = await app.firstWindow();
  await win.emulateMedia({ colorScheme: "light" }); // the OS is light; the switch must override it

  await win.getByRole("button", { name: "Dark theme" }).first().click();
  await expect.poll(() => background(win)).toBe("rgb(6, 11, 20)");
  await win.getByRole("button", { name: "New analysis" }).click();
  await win.fill("#f-path", makeCsv(dir));
  await win.getByRole("button", { name: "Run", exact: true }).click();
  await win.getByRole("tab", { name: "Spectra" }).click();
  await expect.poll(() => plotText(win)).toBe("#edf4ff");

  // switching in the workspace redraws the open plot in the new palette
  await win.getByRole("button", { name: "Light theme" }).click();
  await expect.poll(() => background(win)).toBe("rgb(255, 255, 255)");
  await expect.poll(() => plotText(win)).toBe("#172337");

  // System follows the OS
  await win.getByRole("button", { name: "System theme" }).click();
  await win.emulateMedia({ colorScheme: "dark" });
  await expect.poll(() => background(win)).toBe("rgb(6, 11, 20)");
  await expect.poll(() => plotText(win)).toBe("#edf4ff");

  // a forced choice survives a restart
  await win.getByRole("button", { name: "Light theme" }).click();
  await app.close();
  app = await electron.launch({ args: ["."], env });
  win = await app.firstWindow();
  await win.emulateMedia({ colorScheme: "dark" });
  await expect(win.getByRole("button", { name: "Light theme" }).first()).toHaveAttribute("aria-pressed", "true");
  await expect.poll(() => background(win)).toBe("rgb(255, 255, 255)");
  await app.close();
});
