// @vitest-environment node
import { expect, it } from "vitest";
import { dataUrlToBuffer } from "./files";

it("decodes base64 PNG data URLs", () => {
  const png = Buffer.from([0x89, 0x50, 0x4e, 0x47]);
  expect(dataUrlToBuffer(`data:image/png;base64,${png.toString("base64")}`)).toEqual(png);
});

it("decodes URI-encoded SVG data URLs", () => {
  const svg = '<svg xmlns="http://www.w3.org/2000/svg"><text>δ 3.05</text></svg>';
  expect(dataUrlToBuffer(`data:image/svg+xml,${encodeURIComponent(svg)}`).toString("utf8")).toBe(svg);
});

it("rejects anything else", () => {
  expect(() => dataUrlToBuffer("/etc/passwd")).toThrow("Not a data URL");
});
