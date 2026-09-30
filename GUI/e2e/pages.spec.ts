import { _electron as electron, expect, test } from "@playwright/test";
import { existsSync, mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

function makeCsv(dir: string): string {
  const ppm = Array.from({ length: 20 }, (_, i) => (10 - i * 0.5).toFixed(4));
  const rows = Array.from({ length: 8 }, (_, i) => [`s${i}`, i < 4 ? "A" : "B", ...ppm.map((_, j) => (1 + ((i * 7 + j) % 5) / 10).toFixed(3))].join(","));
  const file = join(dir, "data.csv");
  writeFileSync(file, [["Sample", "Group", ...ppm].join(","), ...rows].join("\n"));
  return file;
}

const shots = process.env.SHOTS; // optional: save each page at both window sizes for review

async function snap(win: import("@playwright/test").Page, name: string) {
  if (!shots) return;
  mkdirSync(join(shots, "pages"), { recursive: true });
  for (const scheme of ["light", "dark"] as const) {
    await win.emulateMedia({ colorScheme: scheme });
    for (const [w, h] of [[1400, 860], [960, 600]]) {
      await win.setViewportSize({ width: w, height: h });
      await win.screenshot({ path: join(shots, "pages", `${name}-${scheme}-${w}.png`) });
    }
  }
  await win.emulateMedia({ colorScheme: null });
}

test("landing, project management and About & License", async () => {
  const dir = mkdtempSync(join(tmpdir(), "metbit-e2e-"));
  const env = { ...process.env, METBIT_E2E_DIR: dir, METBIT_USER_DATA: join(dir, "user") } as Record<string, string>;
  const app = await electron.launch({ args: ["."], env });
  const win = await app.firstWindow();

  // landing first: no session yet, no projects yet, and the credit in the footer
  await expect(win.getByRole("heading", { name: "metbit", level: 1 })).toBeVisible();
  await expect(win.getByRole("button", { name: "Continue session" })).toHaveCount(0);
  await expect(win.getByText("Projects you save or open appear here.")).toBeVisible();
  await expect(win.locator(".page-foot")).toContainText("Developed by Kawa Technology");

  // work, save, then the landing offers to continue and lists the project
  await win.getByRole("button", { name: "New analysis" }).click();
  await win.fill("#f-path", makeCsv(dir));
  await win.getByRole("button", { name: "Run", exact: true }).click();
  await expect(win.locator("table.data")).toContainText("s7");
  await win.getByRole("button", { name: "Save project…" }).click();
  await expect.poll(() => existsSync(join(dir, "analysis.metbit"))).toBe(true);
  await win.locator("button.rail-brand").click();
  await expect(win.getByRole("button", { name: "Continue session" })).toBeVisible();
  await expect(win.locator(".recent-list")).toContainText("analysis");
  await snap(win, "home");

  // project management: rename, duplicate, search, remove from list
  await win.getByRole("button", { name: "Manage projects →" }).click();
  await expect(win.getByRole("heading", { name: "Projects" })).toBeVisible();
  await win.getByRole("button", { name: "Rename" }).click();
  await win.fill('[aria-label="New name"]', "urine study");
  await win.locator('[aria-label="New name"]').press("Enter");
  await expect(win.locator(".project-table")).toContainText("urine study");
  expect(existsSync(join(dir, "urine study.metbit"))).toBe(true);
  await win.getByRole("button", { name: "Duplicate" }).first().click();
  await expect(win.locator(".project-table tbody tr")).toHaveCount(2);
  expect(existsSync(join(dir, "urine study copy.metbit"))).toBe(true);
  await snap(win, "projects");
  await win.fill('[aria-label="Search projects"]', "copy");
  await expect(win.locator(".project-table tbody tr")).toHaveCount(1);
  await win.getByRole("button", { name: "Remove from list" }).click();
  await win.fill('[aria-label="Search projects"]', "");
  await expect(win.locator(".project-table tbody tr")).toHaveCount(1);
  expect(existsSync(join(dir, "urine study copy.metbit"))).toBe(true); // the file stays

  // open from the list goes to the workspace with the project loaded
  await win.getByRole("button", { name: "Open", exact: true }).click();
  await expect(win.locator(".rail")).toBeVisible();
  await expect(win.getByText("Opened urine study.metbit")).toBeVisible();

  // the last page: About & License
  await win.getByRole("button", { name: "About & License" }).click();
  await expect(win.locator(".license")).toContainText("MIT License");
  await expect(win.locator(".license")).toContainText("Copyright (c) 2024 Theerayut");
  await expect(win.locator(".kawa")).toHaveText("Kawa Technology");
  await expect(win.locator(".notices")).toContainText("Plotly.js");
  await snap(win, "about");
  await win.getByRole("button", { name: "Workspace →" }).click();
  await expect(win.locator(".rail")).toBeVisible();

  await app.close();
});
