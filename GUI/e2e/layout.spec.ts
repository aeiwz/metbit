import { _electron as electron, expect, test, type Page } from "@playwright/test";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

// optional: SHOTS=<dir> saves a screenshot of every audited view for review
const shots = process.env.SHOTS;

// 24 samples in three dose groups, an Age column (PLS-R target) and a few peaks, so every stage has work to do
function makeCsv(dir: string): string {
  const n = 600;
  const ppm = Array.from({ length: n }, (_, j) => (10 - (j * 10) / n).toFixed(4));
  const peak = (j: number, at: number, width: number) => Math.exp(-((j - at) ** 2) / width);
  const rows = Array.from({ length: 24 }, (_, i) => {
    const group = i < 8 ? "Control" : i < 16 ? "Low dose" : "High dose";
    const dose = i < 8 ? 1 : i < 16 ? 1.6 : 2.4;
    const values = Array.from({ length: n }, (_, j) =>
      (peak(j, 120, 8) * dose * 4 + peak(j, 124, 8) * dose * 3 + peak(j, 400, 10) * 5 + peak(j, 480, 6) * (3 - dose) +
        Math.abs(Math.sin(i * 3 + j)) * 0.15 + 0.2).toFixed(5),
    );
    return [`sample_${String(i).padStart(2, "0")}`, group, (30 + i * 1.5 + Math.sin(i) * 2).toFixed(1), ...values].join(",");
  });
  const file = join(dir, "audit.csv");
  writeFileSync(file, [["Sample", "Group", "Age", ...ppm].join(","), ...rows].join("\n"));
  return file;
}

// Measures rendered boxes: plot text (titles, notes, labels, legends, colorbars, axis titles, ticks) must not
// overlap or leave the plot; tab-bar text must stay on one line; rail, panel and tab bar must not overflow.
function audit() {
  const found: string[] = [];
  const R = (el: Element) => el.getBoundingClientRect();
  const area = (a: DOMRect, b: DOMRect) =>
    Math.max(0, Math.min(a.right, b.right) - Math.max(a.left, b.left)) * Math.max(0, Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top));
  const plot = document.querySelector(".plot");
  if (plot) {
    const box = R(plot);
    const items: { kind: string; text: string; r: DOMRect }[] = [];
    const add = (kind: string, sel: string) =>
      plot.querySelectorAll(sel).forEach((el) => {
        const r = R(el);
        if (r.width > 0 && r.height > 0) items.push({ kind, text: (el.textContent || "").trim().slice(0, 40), r });
      });
    add("title", ".g-gtitle text");
    add("legend", ".legend .bg");
    add("colorbar", ".colorbar .cbbg");
    add("note", "g.annotation-text-g");
    add("xtitle", ".g-xtitle text");
    add("ytitle", ".g-ytitle text");
    for (const it of items) {
      const r = it.r;
      if (r.left < box.left - 1 || r.right > box.right + 1 || r.top < box.top - 1 || r.bottom > box.bottom + 1)
        found.push(`clipped ${it.kind} "${it.text}"`);
    }
    for (let i = 0; i < items.length; i++)
      for (let j = i + 1; j < items.length; j++)
        if (area(items[i].r, items[j].r) > 4)
          found.push(`${items[i].kind} "${items[i].text}" overlaps ${items[j].kind} "${items[j].text}"`);
    // y tick labels vs y title
    const yt = plot.querySelector(".g-ytitle text");
    if (yt) plot.querySelectorAll(".ytick text").forEach((t) => { if (area(R(t), R(yt)) > 4) found.push(`y tick "${t.textContent}" overlaps y title`); });
    const xt = plot.querySelector(".g-xtitle text");
    if (xt) plot.querySelectorAll(".xtick text").forEach((t) => { if (area(R(t), R(xt)) > 4) found.push(`x tick "${t.textContent}" overlaps x title`); });
  }
  const tabs = [...document.querySelectorAll(".tabs [role=tab]")].map(R);
  const right = [...document.querySelectorAll(".tabs-right > *")].map(R);
  for (const t of tabs) for (const r of right) if (area(t, r) > 1) found.push("tab overlaps tab-bar action");
  document.querySelectorAll(".tabs [role=tab], .tabs-right > *").forEach((el) => {
    const cs = getComputedStyle(el);
    const inner = el.clientHeight - parseFloat(cs.paddingTop) - parseFloat(cs.paddingBottom);
    if (inner > parseFloat(cs.lineHeight) * 1.5) found.push(`tab bar text wraps: "${(el.textContent || "").trim()}"`);
  });
  const bar = document.querySelector(".tabs");
  if (bar && bar.scrollWidth > bar.clientWidth + 1) found.push("tab bar overflows");
  document.querySelectorAll(".panel").forEach((p) => { if (p.scrollWidth > p.clientWidth + 1) found.push("panel overflows horizontally"); });
  const rail = document.querySelector(".rail");
  if (rail && rail.scrollHeight > rail.clientHeight + 1) found.push("rail overflows vertically");
  const foot = document.querySelector(".rail-footer");
  if (foot && R(foot).bottom > window.innerHeight + 1) found.push("rail footer off screen");
  return [...new Set(found)];
}

const STAGES: { name: string; setup?: (w: Page) => Promise<void> }[] = [
  { name: "Normalize" }, { name: "Align" }, { name: "PCA" },
  { name: "OPLS-DA", setup: async (w) => { await w.selectOption("#f-group_b", "High dose"); await w.fill("#f-permutations", "20"); } },
  { name: "PLS-R" }, { name: "LDA" }, { name: "ICA" }, { name: "HCA" },
  { name: "STOCSY", setup: async (w) => { await w.fill("#f-anchor_ppm", "8"); } },
  { name: "Volcano", setup: async (w) => { await w.selectOption("#f-group_b", "High dose"); } },
  { name: "Group tests", setup: async (w) => { await w.fill("#f-bin_width", "0.05"); } },
  { name: "Annotate" },
];

test("no overlapping, clipped or wrapped content in any view at large and minimum window sizes", async () => {
  test.setTimeout(300_000); // runs every stage and visits every view twice (about 1.5 minutes)
  const dir = mkdtempSync(join(tmpdir(), "metbit-layout-"));
  if (shots) mkdirSync(join(shots, "audit"), { recursive: true });
  const report: Record<string, string[]> = {};
  const app = await electron.launch({ args: ["."], env: { ...process.env, METBIT_USER_DATA: join(dir, "user") } as Record<string, string> });
  const win = await app.firstWindow();
  const run = () => win.getByRole("button", { name: "Run", exact: true }).click();
  const check = async (key: string) => {
    await win.waitForTimeout(900);
    const found = await win.evaluate(audit);
    if (found.length) report[key] = found;
    if (shots) await win.screenshot({ path: join(shots, "audit", `${key.replace(/[^\w-]+/g, "_")}.png`) });
  };
  const views = async (vp: string, name: string) => {
    await win.locator(".rail button", { hasText: name }).first().click();
    await win.waitForTimeout(400);
    const tabs = await win.locator(".tabs [role=tab]").allTextContents();
    for (const t of tabs) {
      await win.getByRole("tab", { name: t, exact: true }).click();
      await check(`${vp}-${name}-${t}`);
    }
  };
  await win.setViewportSize({ width: 1400, height: 860 });
  await check("L-import-guide");
  await win.fill("#f-path", makeCsv(dir)); await run();
  await expect(win.locator("table.data")).toContainText("sample_01");
  for (const s of STAGES) {
    await win.locator(".rail button", { hasText: s.name }).first().click();
    if (s.setup) await s.setup(win);
    await run();
    await expect(win.locator(".tabs [role=tab]").first()).toBeVisible({ timeout: 120000 });
    await win.waitForTimeout(500);
  }
  // two labels on the annotate spectra so label overlays are audited everywhere
  await win.locator(".rail button", { hasText: "Annotate" }).click();
  await expect(win.locator(".plot .main-svg").first()).toBeVisible();
  const box = (await win.locator(".plot .nsewdrag").first().boundingBox())!;
  for (const [fx, fy, label] of [[0.2, 0.2, "aromatic"], [0.66, 0.6, "creatine"]] as const) {
    await win.mouse.move(box.x + box.width * fx, box.y + box.height * fy);
    await win.mouse.click(box.x + box.width * fx, box.y + box.height * fy);
    await win.getByLabel("Label", { exact: true }).fill(label);
    await win.getByLabel("Label", { exact: true }).press("Enter");
    await win.waitForTimeout(500);
  }
  for (const vp of [["L", 1400, 860], ["S", 960, 600]] as const) {
    await win.setViewportSize({ width: vp[1], height: vp[2] });
    for (const name of ["Import", ...STAGES.map((s) => s.name)]) await views(vp[0], name);
  }
  await win.getByRole("button", { name: "Plot style…" }).click();
  await check("S-style-panel");
  await app.close();
  expect(report).toEqual({}); // view -> problems found; empty means every view is clean
});
