import { render } from "preact";
import { expect, it } from "vitest";
import { BrandMark, Lockup, MARK_PATH } from "./Brand";

it("draws the identity's resolved-m mark exactly, with the sampled-variable square", () => {
  const root = document.createElement("div");
  render(<BrandMark size={32} />, root);
  const svg = root.querySelector("svg")!;
  expect(svg.getAttribute("viewBox")).toBe("0 0 172 120");
  expect(svg.getAttribute("aria-hidden")).toBe("true");
  expect(root.querySelector("path")!.getAttribute("d")).toBe(MARK_PATH);
  expect(MARK_PATH).toBe("M 12 92 H 27 C 43 92 35 26 52 26 C 69 26 61 92 76 92 C 91 92 82 42 99 42 C 116 42 107 92 125 92 H 143");
  expect(root.querySelector("rect")!.getAttribute("x")).toBe("150");
  expect(svg.getAttribute("width")).toBe("32");
});

it("drops the square below 32 px, as the guidelines ask", () => {
  const root = document.createElement("div");
  render(<BrandMark size={24} />, root);
  expect(root.querySelector("rect")).toBeNull();
});

it("locks up the lowercase name with the quiet GUI descriptor", () => {
  const root = document.createElement("div");
  render(<Lockup gui />, root);
  expect(root.querySelector(".lockup-name")!.textContent).toBe("metbit");
  expect(root.querySelector(".lockup-gui")!.textContent).toBe("GUI");
  render(<Lockup />, root);
  expect(root.querySelector(".lockup-gui")).toBeNull();
});
