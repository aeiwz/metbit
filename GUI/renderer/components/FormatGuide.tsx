import { Fragment, type ComponentChildren } from "preact";

interface Props {
  source: string;
  onSaveExample: () => void;
}

export function FormatGuide({ source, onSaveExample }: Props) {
  return source === "bruker" ? <BrukerGuide /> : <CsvGuide onSaveExample={onSaveExample} />;
}

function Steps({ items }: { items: ComponentChildren[] }) {
  return (
    <ol class="steps">
      {items.map((item, i) => (
        <li key={i}>{item}</li>
      ))}
    </ol>
  );
}

// Processed data table -------------------------------------------------------

const META = ["Sample", "Group", "Age"];
const SPEC = ["10.000", "9.997", "9.994", "…", "0.000"];
const ROWS = [
  ["s01", "Control", "34", "0.0123", "0.0119", "0.0121", "…", "0.0045"],
  ["s02", "Treated", "51", "0.0131", "0.0127", "0.0125", "…", "0.0051"],
  ["s03", "Control", "47", "0.0118", "0.0122", "0.0117", "…", "0.0043"],
];

const CSV_STEPS = [
  <>
    One row per sample, one column per chemical shift. Spectral headers are the ppm values as plain numbers (for
    example <code>9.997</code>); any header that reads as a number is treated as a spectral column.
  </>,
  <>Metadata columns (sample ID, group, other factors) have text headers and can sit anywhere, usually first.</>,
  <>
    Include a group column and enter its name in <b>Group column</b>. Numeric metadata such as age or dose can later
    be a PLS-R target.
  </>,
  <>Spectra must already be processed: phased, baseline corrected and calibrated (TSP / DSS at 0 ppm). Binned data is fine.</>,
  <>No missing values. Remove unwanted regions (for example water) by deleting those columns.</>,
  <>
    Save as a comma-separated UTF-8 CSV. From Python:{" "}
    <code>pd.concat([meta, spectra], axis=1).to_csv("data.csv", index=False)</code>. Export CSV on any data stage in
    this app writes the same format.
  </>,
];

function CsvGuide({ onSaveExample }: { onSaveExample: () => void }) {
  return (
    <section class="guide" aria-label="File format">
      <h3>Processed data table (CSV)</h3>
      <p class="lede">Samples in rows. Metadata columns, then spectral columns with ppm values as headers.</p>
      <div class="mock-scroll">
        <table class="mock-table">
          <thead>
            <tr class="bands">
              <th colSpan={META.length} class="meta">
                metadata
              </th>
              <th colSpan={SPEC.length} class="spec">
                spectral columns (ppm)
              </th>
            </tr>
            <tr>
              {META.map((h) => (
                <th key={h} class="meta">
                  {h}
                </th>
              ))}
              {SPEC.map((h) => (
                <th key={h} class="spec">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {ROWS.map((row) => (
              <tr key={row[0]}>
                {row.map((cell, i) => (
                  <td key={i} class={i < META.length ? "meta" : "spec"}>
                    {cell}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <h4>How to prepare</h4>
      <Steps items={CSV_STEPS} />
      <div class="mock-actions">
        <button type="button" class="btn" onClick={onSaveExample}>
          Save example CSV
        </button>
        <span class="lede">12 samples, 2 groups, 400 ppm points. Saved and filled in for you.</span>
      </div>
    </section>
  );
}

// Raw Bruker + metadata --------------------------------------------------------

const TREE: { depth: number; name: string; exp?: string }[] = [
  { depth: 0, name: "study/" },
  { depth: 1, name: "61020601/", exp: "61020601" },
  { depth: 2, name: "fid" },
  { depth: 2, name: "acqus" },
  { depth: 2, name: "pdata/1/procs" },
  { depth: 1, name: "61030601/", exp: "61030601" },
  { depth: 2, name: "fid, acqus, pdata/…" },
];

const META_ROWS: Record<string, string[]> = { "61020601": ["sham", "6"], "61030601": ["nx", "6"] };

const BRUKER_STEPS = [
  <>
    Export each experiment from TopSpin as its own folder containing <code>fid</code>, <code>acqus</code> and{" "}
    <code>pdata/1/procs</code>. Process reads the stored phase (PHC0 / PHC1) and the spectrum offset from{" "}
    <code>procs</code>.
  </>,
  <>
    Put all experiments of one study under a single folder, either directly (<code>study/&lt;exp&gt;/fid</code>) or
    one level deeper (<code>study/&lt;batch&gt;/&lt;exp&gt;/fid</code>).
  </>,
  <>Acquire every experiment with the same sweep width and settings: spectra are stacked by position, so point counts must match.</>,
  <>Keep an internal reference (TSP or DSS) in each sample for automatic calibration to 0 ppm.</>,
  <>
    Make a metadata CSV with one row per experiment: a column holding the exact folder names (enter its name in{" "}
    <b>Sample column</b>, for example <code>fid</code>), a group column, and any other factors. Every experiment needs
    a row; extra rows are ignored.
  </>,
];

function BrukerGuide() {
  return (
    <section class="guide" aria-label="File format">
      <h3>Raw Bruker + metadata</h3>
      <p class="lede">A folder of experiments, plus a CSV that names each experiment and its group.</p>
      <div class="mock-files">
        <span class="caption">Bruker folder</span>
        <span />
        <span class="caption">metadata.csv</span>
        <span />
        <span />
        <span class="cells head">
          <span>fid</span>
          <span>Group</span>
          <span>Week</span>
        </span>
        {TREE.map((row) => (
          <Fragment key={row.name}>
            <span style={{ paddingLeft: `${row.depth * 16}px` }} class={row.exp ? "link" : undefined}>
              {row.name}
            </span>
            <span class="arrow">{row.exp ? "→" : ""}</span>
            {row.exp ? (
              <span class="cells">
                <span class="link">{row.exp}</span>
                {META_ROWS[row.exp].map((c, i) => (
                  <span key={i}>{c}</span>
                ))}
              </span>
            ) : (
              <span />
            )}
          </Fragment>
        ))}
      </div>
      <h4>How to prepare</h4>
      <Steps items={BRUKER_STEPS} />
    </section>
  );
}
