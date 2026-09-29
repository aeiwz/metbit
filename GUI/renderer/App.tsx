import Plotly from "plotly.js-dist-min";
import { useEffect, useState } from "preact/hooks";
import { Api, ApiError, type StageInfo, type TablePage } from "./api";
import { DataTable } from "./components/DataTable";
import { JobBar } from "./components/JobBar";
import { ParamPanel, fieldErrors, type FieldErrors, type Values } from "./components/ParamPanel";
import { PlotView, type PickPoint } from "./components/PlotView";
import { Rail } from "./components/Rail";

interface Failure {
  message: string;
  traceback?: string;
}

const LABELS: Record<string, string> = { vip: "VIP", "s-plot": "S-plot", predicted: "Predicted vs actual" };
const label = (t: string) => LABELS[t] ?? t[0].toUpperCase() + t.slice(1);
const basename = (p: string) => p.split(/[\\/]/).pop() ?? p;
const PROJECT = [{ name: "metbit project", extensions: ["metbit"] }];
const IMAGE = { width: 1600, height: 1000 };

export function App() {
  const [api, setApi] = useState<Api | null>(null);
  const [stages, setStages] = useState<StageInfo[]>([]);
  const [current, setCurrent] = useState("import");
  const [tab, setTab] = useState("table");
  const [version, setVersion] = useState(0); // bumps after every run to refetch views
  const [formKey, setFormKey] = useState(0); // bumps when a project replaces the params
  const [figure, setFigure] = useState<any>(null);
  const [page, setPage] = useState<TablePage | null>(null);
  const [pct, setPct] = useState<number | null>(null);
  const [errors, setErrors] = useState<FieldErrors>({});
  const [failure, setFailure] = useState<Failure | null>(null);
  const [backendDown, setBackendDown] = useState(false);
  const [picked, setPicked] = useState<number | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const stage = stages.find((s) => s.id === current);
  const hasResult = stage?.status === "done" || stage?.status === "stale";
  const tabs = stage ? [...stage.views, ...(stage.has_table ? ["table"] : [])] : [];
  const pickMode = stage?.ppm_views[tab];
  const onPick = pickMode
    ? (pt: PickPoint) => {
        const v = Number(pickMode === "customdata" ? (pt.customdata as unknown[] | undefined)?.[0] : pt.x);
        if (Number.isFinite(v)) setPicked(v);
      }
    : undefined;

  useEffect(() => {
    Api.connect().then(setApi);
    window.metbit.onBackendExit(() => setBackendDown(true));
  }, []);

  useEffect(() => {
    api?.stages().then(setStages);
  }, [api]);

  useEffect(() => {
    setFigure(null);
    setPage(null);
    if (!api || !hasResult) return;
    let live = true; // drop responses for a view the user already left
    const load = tab === "table" ? api.table(current).then((p) => live && setPage(p)) : api.figure(current, tab).then((f) => live && setFigure(f));
    load.catch((e: Error) => live && setFailure({ message: e.message }));
    return () => {
      live = false;
    };
  }, [api, current, tab, hasResult, version]);

  function flash(message: string) {
    setNotice(message);
    setTimeout(() => setNotice((n) => (n === message ? null : n)), 4000);
  }

  function select(id: string) {
    setCurrent(id);
    setErrors({});
    setFailure(null);
    setTab(stages.find((s) => s.id === id)?.views[0] ?? "table");
  }

  async function refresh(client: Api) {
    setStages(await client.stages());
    setVersion((v) => v + 1);
    setFormKey((k) => k + 1);
  }

  async function guarded(action: () => Promise<void>) {
    try {
      await action();
    } catch (e) {
      setFailure({ message: (e as Error).message });
    }
  }

  async function run(values: Values) {
    if (!api) return;
    setErrors({});
    setFailure(null);
    setPct(0.05);
    setStages((ss) => ss.map((s) => (s.id === current ? { ...s, status: "running" } : s)));
    try {
      await api.run(current, values, (e) => {
        if (e.event === "progress") setPct(e.data.pct);
        if (e.event === "error") setFailure(e.data);
      });
    } catch (e) {
      if (e instanceof ApiError && e.status === 422) setErrors(fieldErrors(e.detail));
      else setFailure({ message: (e as Error).message });
    }
    setPct(null);
    setStages(await api.stages());
    setVersion((v) => v + 1);
  }

  const exportCsv = () =>
    guarded(async () => {
      const path = await window.metbit.saveDialog(`${current}.csv`, [{ name: "CSV", extensions: ["csv"] }]);
      if (!path || !api) return;
      await api.exportCsv(current, path);
      flash(`Saved ${basename(path)}`);
    });

  const exportImage = (format: "png" | "svg") =>
    guarded(async () => {
      if (!figure) return;
      // the original metbit figure, not the themed on-screen one, on an opaque white page so it
      // stays readable in dark viewers; extra top margin fits metbit's Q2 / R2 annotations
      const layout = {
        ...figure.layout,
        ...IMAGE,
        paper_bgcolor: "white",
        plot_bgcolor: "white",
        margin: { ...figure.layout.margin, t: Math.max(figure.layout.margin?.t ?? 0, 140) },
      };
      const url = await Plotly.toImage({ data: figure.data, layout }, { format, ...IMAGE });
      const path = await window.metbit.saveImage(`${current}-${tab}.${format}`, url);
      if (path) flash(`Saved ${basename(path)}`);
    });

  const saveProject = () =>
    guarded(async () => {
      const path = await window.metbit.saveDialog("analysis.metbit", PROJECT);
      if (!path || !api) return;
      await api.saveProject(path);
      flash(`Saved ${basename(path)}`);
    });

  const openProject = () =>
    guarded(async () => {
      const path = await window.metbit.openFile(PROJECT);
      if (!path || !api) return;
      await api.loadProject(path);
      await refresh(api);
      select("import");
      flash(`Opened ${basename(path)}`);
    });

  async function restart() {
    await window.metbit.restartBackend();
    setBackendDown(false);
    const next = await Api.connect();
    const { restored } = await next.restore(); // before setApi, so the stage list reflects it
    setApi(next);
    setFormKey((k) => k + 1);
    select("import");
    if (restored) flash("Session restored");
  }

  if (!stage) return <div class="placeholder">Starting metbit…</div>;

  let body;
  if (failure) {
    body = (
      <div class="placeholder">
        <div>
          <p class="error">{failure.message}</p>
          {failure.traceback && (
            <details>
              <summary>Details</summary>
              <pre>{failure.traceback}</pre>
            </details>
          )}
        </div>
      </div>
    );
  } else if (!hasResult) {
    body = <div class="placeholder">{stage.status === "running" ? "Running…" : "Set parameters and press Run"}</div>;
  } else if (tab === "table") {
    body = page && <DataTable page={page} />;
  } else {
    body = figure && <PlotView figure={figure} onPick={onPick} />;
  }

  return (
    <div class="layout">
      <JobBar pct={pct} />
      <Rail stages={stages} current={current} onSelect={select} onOpen={openProject} onSave={saveProject} />
      <main class="center">
        <div class="tabs" role="tablist">
          {hasResult &&
            tabs.map((t) => (
              <button key={t} role="tab" aria-selected={t === tab} onClick={() => setTab(t)}>
                {label(t)}
              </button>
            ))}
          <div class="tabs-right">
            {notice && (
              <span class="notice" aria-live="polite">
                {notice}
              </span>
            )}
            {picked !== null && <span class="picked">δ {picked.toFixed(3)} ppm</span>}
            {hasResult && stage.exportable && (
              <button class="link" onClick={exportCsv}>
                Export CSV
              </button>
            )}
            {hasResult && tab !== "table" && figure && (
              <>
                <button class="link" onClick={() => exportImage("png")}>
                  PNG
                </button>
                <button class="link" onClick={() => exportImage("svg")}>
                  SVG
                </button>
              </>
            )}
          </div>
        </div>
        <div class="view">{body}</div>
      </main>
      <ParamPanel
        key={`${current}:${formKey}`}
        title={stage.title}
        schema={stage.schema}
        initial={stage.params}
        errors={errors}
        busy={pct !== null}
        onRun={run}
        picked={picked}
      />
      {backendDown && (
        <div class="banner" role="alert">
          Backend stopped
          <button class="btn" onClick={restart}>
            Restart
          </button>
        </div>
      )}
    </div>
  );
}
