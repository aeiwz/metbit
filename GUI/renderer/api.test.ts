import { expect, it } from "vitest";
import { parseSse } from "./api";

it("parses complete events and keeps the partial remainder", () => {
  const buf = 'event: progress\ndata: {"pct":0.5}\n\nevent: done\ndata: {"stage":"pca"}\n\nevent: err';
  const { events, rest } = parseSse(buf);
  expect(events).toEqual([
    { event: "progress", data: { pct: 0.5 } },
    { event: "done", data: { stage: "pca" } },
  ]);
  expect(rest).toBe("event: err");
});

it("returns nothing for an empty buffer", () => {
  expect(parseSse("")).toEqual({ events: [], rest: "" });
});
