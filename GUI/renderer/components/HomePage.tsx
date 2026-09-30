import { fmtDate, type Page } from "./PageNav";

interface Props {
  version: string;
  recent: ProjectEntry[];
  hasSession: boolean; // an analysis is loaded (restored or in progress)
  onNew: () => void;
  onContinue: () => void;
  onOpenFile: () => void;
  onOpenPath: (path: string) => void;
  onNavigate: (page: Page) => void;
}

const STEPS = [
  { title: "Import & process", text: "CSV tables or raw Bruker FIDs, then normalize and align." },
  { title: "Explore", text: "PCA, OPLS-DA, PLS-R, LDA, ICA and HCA." },
  { title: "Test", text: "Volcano, group tests, STOCSY and peak annotation." },
  { title: "Time series", text: "Trajectories, time profiles, ASCA and mixed models." },
];

// benzene mark, like the loading animation
const HEX = Array.from({ length: 6 }, (_, k) => {
  const a = (Math.PI / 3) * k - Math.PI / 2;
  return `${50 + 34 * Math.cos(a)},${50 + 34 * Math.sin(a)}`;
}).join(" ");

export function HomePage({ version, recent, hasSession, onNew, onContinue, onOpenFile, onOpenPath, onNavigate }: Props) {
  return (
    <div class="home">
      <section class="hero">
        <svg class="hero-mark" viewBox="0 0 100 100" aria-hidden="true">
          <polygon points={HEX} />
          <circle cx="50" cy="50" r="18" />
        </svg>
        <div>
          <h1>metbit</h1>
          <p class="lead">NMR metabolomics, from raw spectra to biomarkers.</p>
          <div class="row hero-actions">
            <button type="button" class="btn primary" onClick={onNew}>
              New analysis
            </button>
            <button type="button" class="btn" onClick={onOpenFile}>
              Open project…
            </button>
            {hasSession && (
              <button type="button" class="btn" onClick={onContinue}>
                Continue session
              </button>
            )}
          </div>
        </div>
      </section>

      <section class="steps" aria-label="Pipeline">
        {STEPS.map((s, i) => (
          <div key={s.title} class="step">
            <span class="step-n">{i + 1}</span>
            <h3>{s.title}</h3>
            <p>{s.text}</p>
          </div>
        ))}
      </section>

      <section class="recent">
        <div class="section-head">
          <h2>Recent projects</h2>
          <button type="button" class="link" onClick={() => onNavigate("projects")}>
            Manage projects →
          </button>
        </div>
        {recent.length === 0 ? (
          <p class="hint">Projects you save or open appear here.</p>
        ) : (
          <ul class="recent-list">
            {recent.slice(0, 5).map((p) => (
              <li key={p.path}>
                <button type="button" class="recent-open" disabled={!p.exists} title={p.path} onClick={() => onOpenPath(p.path)}>
                  <span class="recent-name">{p.name}</span>
                  <span class="hint">{p.exists ? p.folder : "File not found"}</span>
                </button>
                <span class="hint">{fmtDate(p.modifiedAt ?? p.openedAt)}</span>
              </li>
            ))}
          </ul>
        )}
      </section>

      <footer class="page-foot">
        metbit {version} · Developed by Kawa Technology · MIT License ·{" "}
        <button type="button" class="link" onClick={() => onNavigate("about")}>
          About & License
        </button>
      </footer>
    </div>
  );
}
