import { render } from "preact";
import { useState } from "preact/hooks";
import { expect, it, vi } from "vitest";
import { DecimalInput } from "./DecimalInput";

const tick = () => new Promise((r) => setTimeout(r));

// a parent that stores what the input reports, as the forms do
function Host({ live, initial, spy }: { live: boolean; initial: number; spy: (v: unknown) => void }) {
  const [value, setValue] = useState<number | string>(initial);
  return (
    <DecimalInput
      id="d"
      value={value}
      live={live}
      onValue={(v) => {
        spy(v);
        setValue(v);
      }}
    />
  );
}

async function typeEach(input: HTMLInputElement, text: string) {
  input.value = "";
  for (const ch of text) {
    input.value += ch;
    input.dispatchEvent(new Event("input"));
    await tick();
    expect(input.value).toBe(text.slice(0, input.value.length)); // nothing typed is rewritten
  }
}

it("keeps every keystroke of a decimal while reporting numbers live", async () => {
  const root = document.createElement("div");
  const spy = vi.fn();
  render(<Host live initial={0.01} spy={spy} />, root);
  const input = root.querySelector<HTMLInputElement>("#d")!;
  expect(input.value).toBe("0.01");
  for (const text of ["0.05", "0.10", "-0.5", "1e-7", ".25"]) {
    await typeEach(input, text);
    expect(spy).toHaveBeenLastCalledWith(Number(text));
  }
  await typeEach(input, "1e-");
  expect(spy).toHaveBeenLastCalledWith("1e-"); // unfinished: left for the server to explain
});

it("commits on change only when not live, and shows a new outside value", async () => {
  const root = document.createElement("div");
  const onValue = vi.fn();
  const view = (value: number) => render(<DecimalInput id="d" value={value} onValue={onValue} />, root);
  view(0.8);
  const input = root.querySelector<HTMLInputElement>("#d")!;
  input.value = "0.75";
  input.dispatchEvent(new Event("input"));
  expect(onValue).not.toHaveBeenCalled();
  input.dispatchEvent(new Event("change"));
  expect(onValue).toHaveBeenCalledWith(0.75);
  view(0.5); // e.g. a reset or a clamp from the server
  await vi.waitFor(() => expect(input.value).toBe("0.5")); // effects run after paint
});
