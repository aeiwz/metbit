# metbit GUI Phase E1 (Molecule Loading Animation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the plain "Starting metbit…" / "Running…" text with a scientific loading animation: a rotating benzene ring with resonance and an atom pulse, a caption, and a progress percentage.

**Architecture:** One renderer component, `MoleculeLoader`, drawn as inline SVG and animated with CSS keyframes (no dependency). `App` shows it at startup, while a stage runs (with real progress), while a project opens, and while a session restores. Spec: `GUI/docs/specs/2026-09-29-metbit-gui-style-annotation-stocsy-design.md` (section 5).

**Tech Stack:** Preact, SVG, CSS, Vitest, Playwright.

## Global Constraints

- All GUI code lives under `GUI/`.
- Never use the em dash character or the section sign anywhere.
- Commit messages carry no `Co-Authored-By` trailer.
- Keep every source file under 500 lines.
- Accent color and tokens from `theme.css` only; works in light and dark.
- `prefers-reduced-motion: reduce` disables every loader animation.
- The ring turns once per 8 s; the loader root has `role="status"`.
- The percentage shows only after real progress (`pct > 0.05`; 0.05 is the placeholder set when a run starts).
- Renderer: `cd GUI && npm run typecheck && npm test`. E2E: `cd GUI && npm run e2e`.

---

### Task 1: MoleculeLoader component

**Files:**
- Create: `GUI/renderer/components/MoleculeLoader.tsx`, `GUI/renderer/components/MoleculeLoader.test.tsx`
- Modify: `GUI/renderer/theme.css` (append)

**Interfaces:**
- Produces: `MoleculeLoader({ caption: string; progress?: number | null })`; root `div.loader[role=status]`; SVG `svg.molecule` with 6 `line.bond`, 6 `line.double` (alternate ones also `.alt`), 6 `circle.atom`; `span.loader-caption`; `span.loader-progress` with `Math.round(progress * 100)%` only when `progress` is a number.

- [ ] **Step 1: Write the failing test**

Create `GUI/renderer/components/MoleculeLoader.test.tsx`:

```tsx
import { render } from "preact";
import { expect, it } from "vitest";
import { MoleculeLoader } from "./MoleculeLoader";

function mount(progress?: number | null) {
  const root = document.createElement("div");
  render(<MoleculeLoader caption="Running OPLS-DA…" progress={progress} />, root);
  return root;
}

it("draws a benzene ring with alternating double bonds", () => {
  const root = mount();
  expect(root.querySelector(".loader")!.getAttribute("role")).toBe("status");
  expect(root.querySelectorAll(".molecule .atom")).toHaveLength(6);
  expect(root.querySelectorAll(".molecule .bond")).toHaveLength(6);
  expect(root.querySelectorAll(".molecule .double")).toHaveLength(6);
  expect(root.querySelectorAll(".molecule .double.alt")).toHaveLength(3);
  expect(root.querySelector(".loader-caption")!.textContent).toBe("Running OPLS-DA…");
});

it("shows a percentage only when there is progress", () => {
  expect(mount(0.42).querySelector(".loader-progress")!.textContent).toBe("42%");
  expect(mount(null).querySelector(".loader-progress")).toBeNull();
  expect(mount().querySelector(".loader-progress")).toBeNull();
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd GUI && npx vitest run renderer/components/MoleculeLoader.test.tsx`
Expected: FAIL, cannot resolve `./MoleculeLoader`.

- [ ] **Step 3: Implement**

Create `GUI/renderer/components/MoleculeLoader.tsx`:

```tsx
// Benzene ring in a 100 x 100 view box: six atoms on a hexagon, six sigma bonds, and six inner
// double-bond strokes in two alternating Kekule sets that cross-fade (resonance).
const CENTER = 50;
const RADIUS = 30;

const ATOMS = Array.from({ length: 6 }, (_, k) => {
  const a = (Math.PI / 3) * k - Math.PI / 2;
  return [CENTER + RADIUS * Math.cos(a), CENTER + RADIUS * Math.sin(a)] as const;
});

// inner stroke for bond k: pulled toward the center, trimmed at both ends
function innerBond(k: number): [number, number, number, number] {
  const pull = (v: number) => CENTER + (v - CENTER) * 0.76;
  const [x1, y1] = ATOMS[k];
  const [x2, y2] = ATOMS[(k + 1) % 6];
  const ax = pull(x1), ay = pull(y1), bx = pull(x2), by = pull(y2);
  const trim = 0.18;
  return [ax + (bx - ax) * trim, ay + (by - ay) * trim, bx - (bx - ax) * trim, by - (by - ay) * trim];
}

interface Props {
  caption: string;
  progress?: number | null;
}

export function MoleculeLoader({ caption, progress = null }: Props) {
  return (
    <div class="loader" role="status" aria-live="polite">
      <svg class="molecule" viewBox="0 0 100 100" aria-hidden="true">
        <g class="ring">
          {ATOMS.map(([x1, y1], k) => {
            const [x2, y2] = ATOMS[(k + 1) % 6];
            return <line key={`b${k}`} class="bond" x1={x1} y1={y1} x2={x2} y2={y2} />;
          })}
          {ATOMS.map((_, k) => {
            const [x1, y1, x2, y2] = innerBond(k);
            return <line key={`d${k}`} class={`double${k % 2 ? " alt" : ""}`} x1={x1} y1={y1} x2={x2} y2={y2} />;
          })}
          {ATOMS.map(([cx, cy], k) => (
            <circle key={`a${k}`} class="atom" cx={cx} cy={cy} r={4.5} style={{ animationDelay: `${k * 0.25}s` }} />
          ))}
        </g>
      </svg>
      <span class="loader-caption">{caption}</span>
      {typeof progress === "number" && <span class="loader-progress">{Math.round(progress * 100)}%</span>}
    </div>
  );
}
```

Append to `GUI/renderer/theme.css`:

```css
.loader { display: flex; flex-direction: column; align-items: center; gap: var(--s1); color: var(--muted); }
.molecule { width: 96px; height: 96px; overflow: visible; }
.molecule .ring { transform-box: view-box; transform-origin: 50% 50%; animation: ring-turn 8s linear infinite; }
.molecule .bond { stroke: var(--accent); stroke-width: 2; stroke-linecap: round; opacity: 0.55; }
.molecule .double { stroke: var(--accent); stroke-width: 2; stroke-linecap: round; animation: resonance 2.4s ease-in-out infinite; }
.molecule .double.alt { animation-delay: -1.2s; }
.molecule .atom { fill: var(--bg); stroke: var(--accent); stroke-width: 2; animation: atom-pulse 1.5s ease-in-out infinite; }
.loader-caption { font-size: 13px; }
.loader-progress { color: var(--text); font-weight: 500; font-variant-numeric: tabular-nums; }
@keyframes ring-turn { to { transform: rotate(360deg); } }
@keyframes resonance { 0%, 100% { opacity: 0.9; } 50% { opacity: 0.12; } }
@keyframes atom-pulse { 0%, 100% { fill: var(--bg); } 20% { fill: var(--accent); } }
@media (prefers-reduced-motion: reduce) {
  .molecule .ring, .molecule .double, .molecule .atom { animation: none; }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd GUI && npx vitest run renderer/components/MoleculeLoader.test.tsx && npm run typecheck`
Expected: `2 passed`; typecheck exits 0.

- [ ] **Step 5: Commit**

```bash
git add GUI/renderer/components/MoleculeLoader.tsx GUI/renderer/components/MoleculeLoader.test.tsx GUI/renderer/theme.css
git commit -m "feat(gui): add a benzene-ring loading animation"
```

---

### Task 2: Show the loader in the app

**Files:**
- Modify: `GUI/renderer/App.tsx`
- Modify: `GUI/e2e/bruker.spec.ts`

**Interfaces:**
- Consumes: `MoleculeLoader` from Task 1.
- Produces: `App` state `busy: string | null`; captions `Starting metbit…`, `Running <stage title>…`, `Opening project…`, `Restoring session…`.

- [ ] **Step 1: Wire it in**

Run from the repo root:

```bash
python3 - <<'EOF'
import pathlib

p = pathlib.Path("GUI/renderer/App.tsx")
s = p.read_text()
pairs = [
    ('import { JobBar } from "./components/JobBar";\n',
     'import { JobBar } from "./components/JobBar";\nimport { MoleculeLoader } from "./components/MoleculeLoader";\n'),
    ("  const [notice, setNotice] = useState<string | null>(null);\n",
     "  const [notice, setNotice] = useState<string | null>(null);\n"
     "  const [busy, setBusy] = useState<string | null>(null); // caption while a project opens or a session restores\n"),
    ('''      await api.loadProject(path);
      await refresh(api);''',
     '''      setBusy("Opening project…");
      try {
        await api.loadProject(path);
        await refresh(api);
      } finally {
        setBusy(null);
      }'''),
    ('''    await window.metbit.restartBackend();
    setBackendDown(false);
    const next = await Api.connect();
    const { restored } = await next.restore(); // before setApi, so the stage list reflects it''',
     '''    setBusy("Restoring session…");
    let next: Api;
    let restored = false;
    try {
      await window.metbit.restartBackend();
      setBackendDown(false);
      next = await Api.connect();
      ({ restored } = await next.restore()); // before setApi, so the stage list reflects it
    } finally {
      setBusy(null);
    }'''),
    ('  if (!stage) return <div class="placeholder">Starting metbit…</div>;',
     '''  if (!stage)
    return (
      <div class="placeholder">
        <MoleculeLoader caption="Starting metbit…" />
      </div>
    );'''),
    ("  if (guideVisible) {",
     '''  if (busy) {
    body = (
      <div class="placeholder">
        <MoleculeLoader caption={busy} />
      </div>
    );
  } else if (guideVisible) {'''),
    ('''    body = <div class="placeholder">{stage.status === "running" ? "Running…" : "Set parameters and press Run"}</div>;''',
     '''    body =
      stage.status === "running" ? (
        <div class="placeholder">
          {/* 0.05 is the placeholder set when a run starts; show a percentage only after real progress */}
          <MoleculeLoader caption={`Running ${stage.title}…`} progress={pct !== null && pct > 0.05 ? pct : null} />
        </div>
      ) : (
        <div class="placeholder">Set parameters and press Run</div>
      );'''),
]
for a, b in pairs:
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
p.write_text(s)
print("App wired")
EOF
```

- [ ] **Step 2: Assert the loader in the Bruker end-to-end test**

In `GUI/e2e/bruker.spec.ts`, replace:

```ts
  await win.locator(".rail button", { hasText: "Process" }).click();
  await run();
```

with:

```ts
  await win.locator(".rail button", { hasText: "Process" }).click();
  await run();
  // per-sample progress drives the molecule loader's percentage while experiments process
  await expect(win.locator(".loader .loader-caption")).toHaveText("Running Process…");
  await expect(win.locator(".loader-progress")).toBeVisible({ timeout: 10_000 });
```

- [ ] **Step 3: Verify**

Run: `cd GUI && npm run typecheck && npm test && npm run e2e`
Expected: typecheck exits 0; `34 passed`; `8 passed` (the Bruker test skips without MTBLS2052).

- [ ] **Step 4: Commit**

```bash
git add GUI/renderer/App.tsx GUI/e2e/bruker.spec.ts
git commit -m "feat(gui): show the molecule loader at startup, during runs, project open and restore"
```
