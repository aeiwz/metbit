import licenseText from "../../../LICENSE?raw"; // the repository's LICENSE, verbatim
import { Lockup } from "./Brand";

interface Props {
  version: string;
}

// licenses as declared in each package's own metadata (checked against the installed versions)
export const NOTICES: { name: string; use: string; license: string }[] = [
  { name: "Electron", use: "Desktop shell", license: "MIT" },
  { name: "Preact", use: "Interface", license: "MIT" },
  { name: "Plotly.js / Plotly for Python", use: "Charts", license: "MIT" },
  { name: "Inter", use: "Typeface", license: "SIL Open Font License 1.1" },
  { name: "FastAPI", use: "Analysis server", license: "MIT" },
  { name: "Uvicorn", use: "Analysis server", license: "BSD-3-Clause" },
  { name: "Pydantic", use: "Parameter validation", license: "MIT" },
  { name: "NumPy", use: "Numerics", license: "BSD-3-Clause (with bundled permissive licenses)" },
  { name: "pandas", use: "Data tables", license: "BSD-3-Clause" },
  { name: "SciPy", use: "Statistics", license: "BSD-3-Clause" },
  { name: "scikit-learn", use: "Multivariate models", license: "BSD-3-Clause" },
  { name: "statsmodels", use: "Mixed models", license: "BSD-3-Clause" },
  { name: "PyArrow", use: "Project files", license: "Apache-2.0" },
  { name: "nmrglue", use: "Bruker import", license: "BSD-3-Clause" },
  { name: "pybaselines", use: "Baseline correction", license: "BSD-3-Clause" },
  { name: "pyChemometrics", use: "Chemometrics", license: "BSD-3-Clause" },
  { name: "lingress", use: "Regression", license: "MIT" },
];

export function AboutPage({ version }: Props) {
  return (
    <div class="about">
      <section class="credit">
        <div class="about-lockup">
          <Lockup gui size={64} />
        </div>
        <h1>About metbit</h1>
        <p class="lead">NMR metabolomics analysis, from raw spectra to biomarkers.</p>
        <dl class="facts">
          <dt>Version</dt>
          <dd>{version}</dd>
          <dt>Developed by</dt>
          <dd class="kawa">Kawa Technology</dd>
          <dt>metbit library</dt>
          <dd>Theerayut (aeiwz)</dd>
          <dt>License</dt>
          <dd>MIT</dd>
        </dl>
      </section>

      <section>
        <h2>License terms</h2>
        <pre class="license" aria-label="License text">
          {licenseText.trim()}
        </pre>
      </section>

      <section>
        <h2>Open-source components</h2>
        <p class="hint">metbit is built on these projects. Each keeps its own license; the full texts ship with the packages.</p>
        <table class="notices">
          <thead>
            <tr>
              <th>Component</th>
              <th>Used for</th>
              <th>License</th>
            </tr>
          </thead>
          <tbody>
            {NOTICES.map((n) => (
              <tr key={n.name}>
                <td>{n.name}</td>
                <td class="hint">{n.use}</td>
                <td>{n.license}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <footer class="page-foot">
        Developed by Kawa Technology · metbit is free software under the MIT License, provided "as is", without warranty of
        any kind.
      </footer>
    </div>
  );
}
