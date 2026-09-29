import { useEffect, useState } from "preact/hooks";
import type { ComponentChildren } from "preact";
import type { FieldSchema, JsonSchema } from "../api";

export type Values = Record<string, unknown>;
export type FieldErrors = Record<string, string>;

export function defaults(schema: JsonSchema): Values {
  return Object.fromEntries(Object.entries(schema.properties).map(([k, f]) => [k, f.default ?? ""]));
}

export function fieldErrors(detail: unknown): FieldErrors {
  if (!Array.isArray(detail)) return { "": String(detail) };
  return Object.fromEntries(detail.map((d: { loc: unknown[]; msg: string }) => [String(d.loc[0] ?? ""), d.msg]));
}

const visible = (f: FieldSchema, values: Values) =>
  !f.show_if || Object.entries(f.show_if).every(([k, v]) => values[k] === v);

interface Props {
  title: string;
  schema: JsonSchema;
  initial: Values | null;
  errors: FieldErrors;
  busy: boolean;
  onRun: (values: Values) => void;
  picked?: number | null;
  onValuesChange?: (values: Values) => void;
  injected?: { values: Values; n: number } | null; // merged in when n changes (a click set a field)
  children?: ComponentChildren;
}

export function ParamPanel({ title, schema, initial, errors, busy, onRun, picked = null, onValuesChange, children, injected }: Props) {
  const [values, setValues] = useState<Values>(() => ({ ...defaults(schema), ...initial }));
  const set = (k: string, v: unknown) => setValues((s) => ({ ...s, [k]: v }));
  useEffect(() => onValuesChange?.(values), [values]);
  useEffect(() => {
    if (injected) setValues((s) => ({ ...s, ...injected.values }));
  }, [injected?.n]);
  return (
    <div class="panel">
    <form
      class="panel-form"
      onSubmit={(e) => {
        e.preventDefault();
        onRun(values);
      }}
    >
      <h2>{title}</h2>
      {Object.entries(schema.properties)
        .filter(([, f]) => visible(f, values))
        .map(([k, f]) => (
          <Field key={k} name={k} field={f} value={values[k]} error={errors[k]} picked={picked} onChange={(v) => set(k, v)} />
        ))}
      {errors[""] && <p class="error">{errors[""]}</p>}
      <button class="btn primary" type="submit" disabled={busy}>
        {busy ? "Running…" : "Run"}
      </button>
    </form>
    {children}
    </div>
  );
}

interface FieldProps {
  name: string;
  field: FieldSchema;
  value: unknown;
  error?: string;
  picked: number | null;
  onChange: (v: unknown) => void;
}

function Field({ name, field, value, error, picked, onChange }: FieldProps) {
  const id = `f-${name}`;
  const text = (e: Event) => onChange((e.currentTarget as HTMLInputElement).value);
  let control;
  if (field.enum) {
    control = (
      <select id={id} value={String(value)} onChange={text}>
        {field.enum.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    );
  } else if (field.type === "boolean") {
    // onClick, not onChange: click fires for mouse and Space and already carries the new state
    control = <input id={id} type="checkbox" checked={value === true} onClick={(e) => onChange(e.currentTarget.checked)} />;
  } else if (field.type === "integer" || field.type === "number") {
    control = (
      <input
        id={id}
        type="number"
        step={field.type === "integer" ? 1 : "any"}
        min={field.minimum}
        max={field.maximum}
        value={value as number}
        onInput={(e) => {
          const raw = e.currentTarget.value;
          onChange(raw === "" ? "" : Number(raw));
        }}
      />
    );
  } else if (field.format === "file-path" || field.format === "dir-path") {
    const directory = field.format === "dir-path";
    const browse = async () => {
      // ponytail: every file input so far is a CSV; take filters from the schema when that changes
      const path = await window.metbit.openFile(directory ? undefined : [{ name: "CSV", extensions: ["csv"] }], directory || undefined);
      if (path) onChange(path);
    };
    control = (
      <div class="row">
        <input id={id} value={String(value ?? "")} onInput={text} />
        <button type="button" class="btn" onClick={browse}>
          Browse
        </button>
      </div>
    );
  } else {
    control = <input id={id} value={String(value ?? "")} onInput={text} />;
  }
  return (
    <div class={`field${error ? " invalid" : ""}`}>
      <label for={id}>{field.title ?? name}</label>
      {control}
      {field.pick_ppm && picked !== null && (
        <button type="button" class="pick" onClick={() => onChange(picked)}>
          Use {picked.toFixed(3)} ppm
        </button>
      )}
      {field.description && <span class="hint">{field.description}</span>}
      {error && <span class="err">{error}</span>}
    </div>
  );
}
