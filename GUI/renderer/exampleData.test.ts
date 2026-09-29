import { expect, it } from "vitest";
import { makeExampleCsv } from "./exampleData";

it("writes a table the importer accepts", () => {
  const lines = makeExampleCsv().trim().split("\n");
  expect(lines).toHaveLength(13);
  const header = lines[0].split(",");
  expect(header.slice(0, 3)).toEqual(["Sample", "Group", "Age"]);
  const ppm = header.slice(3);
  expect(ppm).toHaveLength(400);
  expect(ppm.every((h) => Number.isFinite(Number(h)))).toBe(true);
  expect(new Set(ppm).size).toBe(400);
  const rows = lines.slice(1).map((l) => l.split(","));
  expect(rows.map((r) => r[1])).toEqual([...Array(6).fill("Control"), ...Array(6).fill("Treated")]);
  expect(rows.every((r) => r.length === 403 && r.slice(2).every((v) => Number.isFinite(Number(v))))).toBe(true);
});

it("is the same every time", () => {
  expect(makeExampleCsv()).toBe(makeExampleCsv());
});
