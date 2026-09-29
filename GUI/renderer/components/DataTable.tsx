import type { TablePage } from "../api";

const fmt = (v: unknown) => (typeof v === "number" && !Number.isInteger(v) ? v.toPrecision(4) : String(v ?? ""));

export function DataTable({ page }: { page: TablePage }) {
  return (
    <>
      <table class="data">
        <thead>
          <tr>
            {page.columns.map((c) => (
              <th key={c}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {page.rows.map((row, i) => (
            <tr key={i}>
              {row.map((v, j) => (
                <td key={j}>{fmt(v)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {page.total > page.rows.length && (
        <p class="muted">
          Showing {page.rows.length} of {page.total}
        </p>
      )}
    </>
  );
}
