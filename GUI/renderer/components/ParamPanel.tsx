import { useState } from "preact/hooks";
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

interface Props {
  title: string;
  schema: JsonSchema;
  initial: Values | null;
  errors: FieldErrors;
  busy: boolean;
  onRun: (values: Values) => void;
}

export function ParamPanel({ title, schema, initial, errors, busy, onRun }: Props) {
  const [values, setValues] = useState<Values>(() => ({ ...defaults(schema), ...initial }));
  const set = (k: string, v: unknown) => setValues((s) => ({ ...s, [k]: v }));
  return (
    <form
      class="panel"
      onSubmit={(e) => {
        e.preventDefault();
        onRun(values);
      }}
    >
      <h2>{title}</h2>
      {Object.entries(schema.properties).map(([k, f]) => (
        <Field key={k} name={k} field={f} value={values[k]} error={errors[k]} onChange={(v) => set(k, v)} />
      ))}
      {errors[""] && <p class="error">{errors[""]}</p>}
      <button class="btn primary" type="submit" disabled={busy}>
        {busy ? "Running…" : "Run"}
      </button>
    </form>
  );
}

interface FieldProps {
  name: string;
  field: FieldSchema;
  value: unknown;
  error?: string;
  onChange: (v: unknown) => void;
}

function Field({ name, field, value, error, onChange }: FieldProps) {
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
  } else if (field.format === "file-path") {
    // ponytail: CSV is the only file input in stage 1; take filters from the schema when Bruker import lands
    const browse = async () => {
      const path = await window.metbit.openFile([{ name: "CSV", extensions: ["csv"] }]);
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
      {error && <span class="err">{error}</span>}
    </div>
  );
}
