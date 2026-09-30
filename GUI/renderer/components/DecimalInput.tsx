import type { JSX } from "preact";
import { useEffect, useState } from "preact/hooks";

const NUMBER = /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/;

// complete numbers (including 1e-7) become numbers; anything else stays as typed for the server to reject
export function toNumber(raw: string): number | string {
  const text = raw.trim();
  return NUMBER.test(text) ? Number(text) : text;
}

const show = (v: unknown) => (v === undefined || v === null ? "" : String(v));

type Props = Omit<JSX.HTMLAttributes<HTMLInputElement>, "value" | "onInput" | "onChange"> & {
  value: unknown;
  onValue: (v: number | string) => void;
  live?: boolean; // report every keystroke (forms read it on submit), else only on change (Enter / leaving the field)
};

/**
 * A decimal field that never rewrites what is being typed. Text, not type="number": converting on each
 * keystroke turned "0." into "0" (so 0.05 could not be typed) and a number input reports "1e-" or "-" as empty.
 */
export function DecimalInput({ value, onValue, live = false, ...rest }: Props) {
  const [draft, setDraft] = useState(show(value));
  // a value from outside (a picked ppm, a reset, a clamp) replaces the draft; our own echo does not
  useEffect(() => {
    if (toNumber(draft) !== value && show(value) !== draft) setDraft(show(value));
  }, [value]);
  return (
    <input
      {...rest}
      type="text"
      inputMode="decimal"
      autoComplete="off"
      spellcheck={false}
      value={draft}
      onInput={(e) => {
        const text = e.currentTarget.value;
        setDraft(text);
        if (live) onValue(toNumber(text));
      }}
      onChange={(e) => {
        if (!live) onValue(toNumber(e.currentTarget.value));
      }}
    />
  );
}
