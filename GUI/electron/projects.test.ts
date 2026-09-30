import { mkdtempSync, readFileSync, writeFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { expect, it } from "vitest";
import { ProjectRegistry, cleanName } from "./projects";

function setup() {
  const dir = mkdtempSync(join(tmpdir(), "metbit-projects-"));
  const project = join(dir, "study.metbit");
  writeFileSync(project, "zip");
  return { dir, project, reg: new ProjectRegistry(join(dir, "projects.json")) };
}

it("remembers projects newest first, with file facts, and survives a restart", async () => {
  const { dir, project, reg } = setup();
  const other = join(dir, "other.metbit");
  writeFileSync(other, "zipzip");
  await reg.add(project);
  await reg.add(other);
  await reg.add(project); // opening again moves it to the top, no duplicate
  const list = await new ProjectRegistry(join(dir, "projects.json")).list();
  expect(list.map((p) => p.name)).toEqual(["study", "other"]);
  expect(list[0]).toMatchObject({ path: project, exists: true, size: 3, folder: dir });
  await reg.remove(other);
  expect((await reg.list()).map((p) => p.name)).toEqual(["study"]);
  expect(existsSync(other)).toBe(true); // removing from the list keeps the file
});

it("marks missing files instead of dropping them", async () => {
  const { reg } = setup();
  await reg.add("/nowhere/gone.metbit");
  expect((await reg.list())[0]).toMatchObject({ name: "gone", exists: false });
});

it("renames and duplicates on disk without overwriting", async () => {
  const { dir, project, reg } = setup();
  await reg.add(project);
  const renamed = await reg.rename(project, "  Urine study  ");
  expect(renamed).toBe(join(dir, "Urine study.metbit"));
  expect(existsSync(project)).toBe(false);
  expect(readFileSync(renamed, "utf8")).toBe("zip");
  writeFileSync(join(dir, "taken.metbit"), "x");
  await expect(reg.rename(renamed, "taken")).rejects.toThrow("already exists");
  const copy = await reg.duplicate(renamed);
  const copy2 = await reg.duplicate(renamed);
  expect([copy, copy2]).toEqual([join(dir, "Urine study copy.metbit"), join(dir, "Urine study copy 2.metbit")]);
  expect((await reg.list()).map((p) => p.name)).toEqual(["Urine study copy 2", "Urine study copy", "Urine study"]);
});

it("only acts on projects it knows, and rejects unsafe names", async () => {
  const { dir, reg } = setup();
  const stranger = join(dir, "stranger.metbit");
  writeFileSync(stranger, "x");
  await expect(reg.rename(stranger, "x")).rejects.toThrow("not in the project list");
  await expect(reg.add(join(dir, "notes.txt"))).rejects.toThrow(".metbit");
  for (const bad of ["", "   ", "../escape", "a/b", "a\\b", "what?", "x".repeat(121)]) expect(() => cleanName(bad)).toThrow();
  expect(cleanName(" ok name ")).toBe("ok name");
});
