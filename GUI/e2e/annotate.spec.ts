import { _electron as electron, expect, test, type Page } from "@playwright/test";
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

const plotLabels = (win: Page) =>
  win.evaluate(() => ((document.querySelector(".plot") as any)?.layout?.annotations ?? []).map((a: any) => a.text));
const labelOffset = (win: Page, text: string) =>
  win.evaluate((t) => ((document.querySelector(".plot") as any)?.layout?.annotations ?? []).find((a: any) => a.text === t)?.ay, text);
const traces = (win: Page) => win.evaluate(() => ((document.querySelector(".plot") as any)?.data ?? []).map((t: any) => t.name));

test("annotator dashboard: add, switch, restyle, drag and export", async () => {
  const dir = mkdtempSync(join(tmpdir(), "metbit-e2e-"));
  const env = { ...process.env, METBIT_E2E_DIR: dir, METBIT_USER_DATA: join(dir, "user") } as Record<string, string>;
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

  // click a peak: current peak pill + focused label field, then Add Label
  const box = (await win.locator(".plot .nsewdrag").first().boundingBox())!;
  await win.mouse.move(box.x + box.width * 0.2, box.y + box.height * 0.3);
  await win.mouse.click(box.x + box.width * 0.2, box.y + box.height * 0.3);
  await expect(win.locator(".peak-pill")).toContainText(" ppm");
  await expect(win.locator("#annotate-label")).toBeFocused();
  await win.fill("#annotate-label", "lactate");
  await win.locator("#annotate-label").press("Enter");
  await expect(win.locator(".label-list")).toContainText("lactate");
  await expect.poll(() => plotLabels(win)).toContain("lactate");

  // median switch: no re-run
  await win.getByRole("button", { name: "Median spectra" }).click();
  await expect.poll(() => traces(win)).toEqual(["A", "B"]);

  // style group B dotted
  await win.locator("summary", { hasText: "Style line" }).click();
  await win.selectOption('[aria-label="Line to style"]', "B");
  await win.selectOption('[aria-label="Line dash"]', "dot");
  await win.getByRole("button", { name: "Apply Style" }).click();
  await expect
    .poll(() => win.evaluate(() => (document.querySelector(".plot") as any).data.find((t: any) => t.name === "B")?.line?.dash))
    .toBe("dot");

  // drag the label; the new offset comes back from the server after a refetch
  // grab the text box: the middle of the whole annotation group is on the arrow, which does not drag
  const label = (await win.locator(".plot g.annotation-text-g", { hasText: "lactate" }).first().boundingBox())!;
  const cx = label.x + label.width / 2;
  const cy = label.y + label.height / 2;
  await win.mouse.move(cx, cy);
  await win.mouse.down();
  await win.mouse.move(cx + 30, cy - 40, { steps: 8 });
  await win.mouse.up();
  await win.getByRole("button", { name: "Single spectra" }).click();
  await expect.poll(() => labelOffset(win, "lactate")).not.toBe(-40);

  // pick the label in the list to edit its text and angle
  const note = (text: string) =>
    win.evaluate((t) => ((document.querySelector(".plot") as any)?.layout?.annotations ?? []).find((a: any) => a.text === t), text);
  await win.locator(".label-list .label-name", { hasText: "lactate" }).click();
  await expect(win.locator(".dash-panel summary").first()).toContainText("Edit annotation");
  await expect(win.locator("#annotate-label")).toHaveValue("lactate");
  await win.fill("#annotate-label", "lactic acid");
  await win.fill('[aria-label="Angle"]', "");
  await win.locator('[aria-label="Angle"]').pressSequentially("-30");
  await expect(win.locator('[aria-label="Angle"]')).toHaveValue("-30");
  await win.getByRole("button", { name: "Save", exact: true }).click();
  await expect.poll(async () => (await note("lactic acid"))?.textangle).toBe(-30);
  await expect(win.locator(".dash-panel summary").first()).toContainText("Add annotation");

  // click the label on the plot to edit it
  const text = (await win.locator(".plot g.annotation-text-g", { hasText: "lactic acid" }).first().boundingBox())!;
  await win.mouse.click(text.x + text.width / 2, text.y + text.height / 2);
  await expect(win.locator("#annotate-label")).toHaveValue("lactic acid");
  await win.fill("#annotate-label", "lactate");
  await win.getByRole("button", { name: "Save", exact: true }).click();
  await expect.poll(async () => (await note("lactate"))?.textangle).toBe(-30);

  // label font and style
  await win.selectOption('[aria-label="Label font"]', "Georgia");
  await win.selectOption('[aria-label="Label style"]', "bold-italic");
  await expect.poll(async () => (await note("lactate"))?.font).toMatchObject({ family: "Georgia", weight: "bold", style: "italic" });

  // self-contained HTML export
  await win.getByRole("button", { name: "Download HTML" }).click();
  const html = join(dir, "annotated-spectra.html");
  await expect.poll(() => existsSync(html)).toBe(true);
  expect(readFileSync(html, "utf8")).toContain("lactate");

  // labels show on other spectra and in the intensity table
  await win.getByRole("tab", { name: "Table" }).click();
  await expect(win.locator("table.data thead")).toContainText("lactate_");
  await win.locator(".rail button", { hasText: "Normalize" }).click();
  await run();
  await plot();
  await expect.poll(() => plotLabels(win)).toContain("lactate");

  await app.close();
});
