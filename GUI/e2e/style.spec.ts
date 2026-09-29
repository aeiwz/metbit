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

// read the rendered Plotly trace for a group straight from the graph div
const groupMarker = (win: Page, group: string, prop: "color" | "size") =>
  win.evaluate(
    ([g, p]) => {
      const gd = document.querySelector(".plot") as any;
      const trace = gd?.data?.find(
        (t: any) => String(t.name).split(",")[0].trim() === g && String(t.mode ?? "markers").includes("markers"),
      );
      return trace?.marker?.[p];
    },
    [group, prop] as const,
  );

test("global style recolors a group; a stage override changes marker size", async () => {
  const dir = mkdtempSync(join(tmpdir(), "metbit-e2e-"));
  const env = { ...process.env, METBIT_USER_DATA: join(dir, "user") } as Record<string, string>;
  const app = await electron.launch({ args: ["."], env });
  const win = await app.firstWindow();
  const run = () => win.getByRole("button", { name: "Run", exact: true }).click();

  await win.fill("#f-path", makeCsv(dir));
  await run();
  await expect(win.locator("table.data")).toContainText("s11");
  await win.locator(".rail button", { hasText: "PCA" }).click();
  await run();
  await expect(win.locator(".plot .main-svg").first()).toBeVisible({ timeout: 30_000 });

  await win.getByRole("button", { name: "Plot style…" }).click();
  await win.getByLabel("B color").fill("#ff0000");
  await expect.poll(() => groupMarker(win, "B", "color")).toBe("#ff0000");
  await win.getByRole("button", { name: "Done" }).click();

  await win.locator("details.style-override summary").click();
  await win.fill("#style-marker_size", "30");
  await win.locator("#style-marker_size").press("Tab"); // number inputs report on change (blur)
  await expect.poll(() => groupMarker(win, "B", "size")).toBe(30);
  await expect(win.locator("details.style-override summary")).toContainText("(custom)");
  await expect(win.locator("details.style-override")).toHaveAttribute("open", ""); // stays open while editing

  await app.close();
});
