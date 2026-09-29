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

const LABELS: Record<string, string> = { vip: "VIP", "s-plot": "S-plot" };
const label = (t: string) => LABELS[t] ?? t[0].toUpperCase() + t.slice(1);

export function App() {
  const [api, setApi] = useState<Api | null>(null);
  const [stages, setStages] = useState<StageInfo[]>([]);
  const [current, setCurrent] = useState("import");
  const [tab, setTab] = useState("table");
  const [version, setVersion] = useState(0); // bumps after every run to refetch views
  const [figure, setFigure] = useState<any>(null);
  const [page, setPage] = useState<TablePage | null>(null);
  const [pct, setPct] = useState<number | null>(null);
  const [errors, setErrors] = useState<FieldErrors>({});
  const [failure, setFailure] = useState<Failure | null>(null);
  const [backendDown, setBackendDown] = useState(false);
  const [picked, setPicked] = useState<number | null>(null);

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

  function select(id: string) {
    setCurrent(id);
    setErrors({});
    setFailure(null);
    setTab(stages.find((s) => s.id === id)?.views[0] ?? "table");
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

  async function restart() {
    await window.metbit.restartBackend();
    setBackendDown(false);
    setApi(await Api.connect());
    select("import");
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
      <Rail stages={stages} current={current} onSelect={select} />
      <main class="center">
        <div class="tabs" role="tablist">
          {hasResult &&
            tabs.map((t) => (
              <button key={t} role="tab" aria-selected={t === tab} onClick={() => setTab(t)}>
                {label(t)}
              </button>
            ))}
          {picked !== null && (
            <span class="picked" aria-live="polite">
              δ {picked.toFixed(3)} ppm
            </span>
          )}
        </div>
        <div class="view">{body}</div>
      </main>
      <ParamPanel
        key={current}
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
