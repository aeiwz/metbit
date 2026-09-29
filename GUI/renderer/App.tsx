import Plotly from "plotly.js-dist-min";
import { useEffect, useState } from "preact/hooks";
import { Api, ApiError, type Annotation, type PlotStyle, type StageInfo, type StageStyle, type StyleOverride, type TablePage } from "./api";
import { DataTable } from "./components/DataTable";
import { FormatGuide } from "./components/FormatGuide";
import { makeExampleCsv } from "./exampleData";
import { JobBar } from "./components/JobBar";
import { MoleculeLoader } from "./components/MoleculeLoader";
import { ParamPanel, fieldErrors, type FieldErrors, type Values } from "./components/ParamPanel";
import { PlotView, type PickPoint } from "./components/PlotView";
import { AnnotationList } from "./components/AnnotationList";
import { AnnotationPopover } from "./components/AnnotationPopover";
import { Rail } from "./components/Rail";
import { StocsyTools, type Anchor } from "./components/StocsyTools";
import { StyleForm, applyPatch } from "./components/StyleForm";

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
  const [busy, setBusy] = useState<string | null>(null); // caption while a project opens or a session restores
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [pending, setPending] = useState<{ ppm: number; x: number; y: number } | null>(null);
  const [injected, setInjected] = useState<{ values: Values; n: number } | null>(null);
  const [recent, setRecent] = useState<Anchor[]>([]); // STOCSY anchors run this session, newest first
  const [importValues, setImportValues] = useState<Values>({});
  const [showGuide, setShowGuide] = useState(false);
  const [sideMode, setSideMode] = useState<"params" | "style">("params");
  const [globalStyle, setGlobalStyle] = useState<PlotStyle | null>(null);
  const [stageStyle, setStageStyle] = useState<StageStyle | null>(null);

  const stage = stages.find((s) => s.id === current);
  const hasResult = stage?.status === "done" || stage?.status === "stale";
  const preViews = stage?.pre_views ?? [];
  const tabs = stage ? [...preViews, ...(hasResult ? [...stage.views, ...(stage.has_table ? ["table"] : [])] : [])] : [];
  const pickMode = stage?.ppm_views[tab];
  const onPick = pickMode
    ? (pt: PickPoint) => {
        const v = Number(pickMode === "customdata" ? (pt.customdata as unknown[] | undefined)?.[0] : pt.x);
        if (!Number.isFinite(v)) return;
        setPicked(v);
        // on the Annotate spectra a click opens the label popover at the click
        if (current === "annotate" && tab === "spectra" && pt.clientX !== undefined && pt.clientY !== undefined) {
          setPending({ ppm: v, x: pt.clientX, y: pt.clientY });
        }
        // on the STOCSY spectra a click sets the anchor, as in metbit's STOCSY app
        if (current === "stocsy" && tab === "spectra") setInjected({ values: { anchor_ppm: Number(v.toFixed(4)) }, n: Date.now() });
      }
    : undefined;

  useEffect(() => {
    Api.connect().then(setApi);
    window.metbit.onBackendExit(() => setBackendDown(true));
  }, []);

  useEffect(() => {
    api?.stages().then(setStages);
    api?.listAnnotations().then(setAnnotations);
  }, [api]);

  useEffect(() => {
    setFigure(null);
    setPage(null);
    if (!api || (!hasResult && !preViews.includes(tab))) return;
    let live = true; // drop responses for a view the user already left
    const load = tab === "table" ? api.table(current).then((p) => live && setPage(p)) : api.figure(current, tab).then((f) => live && setFigure(f));
    load.catch((e: Error) => live && setFailure({ message: e.message }));
    return () => {
      live = false;
    };
  }, [api, current, tab, hasResult, version]);

  // refresh in place (no reset to null): unmounting would collapse the open Style section
  useEffect(() => {
    if (!api || !stages.find((s) => s.id === current)?.views.length) {
      setStageStyle(null);
      return;
    }
    let live = true;
    api.getStageStyle(current).then((st) => live && setStageStyle(st));
    return () => {
      live = false;
    };
  }, [api, current, version, globalStyle, stages.length]);

  function flash(message: string) {
    setNotice(message);
    setTimeout(() => setNotice((n) => (n === message ? null : n)), 4000);
  }

  function select(id: string) {
    setShowGuide(false);
    setSideMode("params");
    setStageStyle(null); // the next stage's style loads fresh
    setPending(null);
    setInjected(null);
    setCurrent(id);
    setErrors({});
    setFailure(null);
    const next = stages.find((s) => s.id === id);
    setTab(next?.pre_views[0] ?? next?.views[0] ?? "table");
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
    const fresh = await api.stages();
    setStages(fresh);
    setVersion((v) => v + 1);
    const ran = fresh.find((s) => s.id === current);
    if (ran?.status === "done") {
      // leave a pre-run view for the result it prepared (STOCSY: spectra to correlation)
      if (ran.pre_views.includes(tab) && ran.views[0]) setTab(ran.views[0]);
      if (current === "stocsy") {
        const a = { anchor_ppm: Number(values.anchor_ppm), p_threshold: Number(values.p_threshold) };
        setRecent((r) => [a, ...r.filter((x) => x.anchor_ppm !== a.anchor_ppm || x.p_threshold !== a.p_threshold)].slice(0, 8));
      }
    }
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

  const saveExample = () =>
    guarded(async () => {
      const url = `data:text/csv;charset=utf-8,${encodeURIComponent(makeExampleCsv())}`;
      const path = await window.metbit.saveImage("metbit-example.csv", url);
      if (!path) return;
      // prefill the Import form so the next step is just Run
      setStages((ss) => ss.map((s) => (s.id === "import" ? { ...s, params: { ...(s.params ?? {}), source: "csv", path } } : s)));
      setFormKey((k) => k + 1);
      flash(`Saved ${basename(path)}`);
    });
  const guideVisible = current === "import" && !failure && (showGuide || (!hasResult && stage?.status !== "running"));

  async function annotationsChanged(client: Api) {
    setAnnotations(await client.listAnnotations());
    setVersion((v) => v + 1); // labels are drawn server-side and feed the intensity table
  }

  const addAnnotation = (label: string, angle: number) =>
    guarded(async () => {
      if (!api || !pending) return;
      await api.addAnnotation(pending.ppm, label, angle);
      setPending(null);
      await annotationsChanged(api);
    });

  const deleteAnnotation = (id: string) =>
    guarded(async () => {
      if (!api) return;
      await api.deleteAnnotation(id);
      await annotationsChanged(api);
    });

  const importAnnotations = () =>
    guarded(async () => {
      const path = await window.metbit.openFile([{ name: "JSON", extensions: ["json"] }]);
      if (!path || !api) return;
      await api.importAnnotations(path);
      await annotationsChanged(api);
      flash(`Imported ${basename(path)}`);
    });

  const exportAnnotations = () =>
    guarded(async () => {
      const path = await window.metbit.saveDialog("annotations.json", [{ name: "JSON", extensions: ["json"] }]);
      if (!path || !api) return;
      await api.exportAnnotations(path);
      flash(`Saved ${basename(path)}`);
    });

  const openStyle = () =>
    guarded(async () => {
      if (!api) return;
      setGlobalStyle(await api.getStyle());
      setSideMode("style");
    });

  const patchGlobal = (patch: StyleOverride) =>
    guarded(async () => {
      if (!api || !globalStyle) return;
      setGlobalStyle(await api.putStyle(applyPatch(globalStyle, patch)));
      setVersion((v) => v + 1);
    });

  const patchStage = (patch: StyleOverride) =>
    guarded(async () => {
      if (!api || !stageStyle) return;
      setStageStyle(await api.putStageStyle(current, applyPatch(stageStyle.override, patch)));
      setVersion((v) => v + 1);
    });

  const resetStage = () =>
    guarded(async () => {
      if (!api) return;
      setStageStyle(await api.putStageStyle(current, {}));
      setVersion((v) => v + 1);
    });

  const openProject = () =>
    guarded(async () => {
      const path = await window.metbit.openFile(PROJECT);
      if (!path || !api) return;
      setBusy("Opening project…");
      try {
        await api.loadProject(path);
        await refresh(api);
      } finally {
        setBusy(null);
      }
      select("import");
      flash(`Opened ${basename(path)}`);
    });

  async function restart() {
    setBusy("Restoring session…");
    let next: Api;
    let restored = false;
    try {
      await window.metbit.restartBackend();
      setBackendDown(false);
      next = await Api.connect();
      ({ restored } = await next.restore()); // before setApi, so the stage list reflects it
    } finally {
      setBusy(null);
    }
    setApi(next);
    setFormKey((k) => k + 1);
    select("import");
    if (restored) flash("Session restored");
  }

  if (!stage)
    return (
      <div class="placeholder">
        <MoleculeLoader caption="Starting metbit…" />
      </div>
    );

  let body;
  if (busy) {
    body = (
      <div class="placeholder">
        <MoleculeLoader caption={busy} />
      </div>
    );
  } else if (guideVisible) {
    body = <FormatGuide source={String(importValues.source ?? "csv")} onSaveExample={saveExample} />;
  } else if (failure) {
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
  } else if (stage.status !== "running" && preViews.includes(tab)) {
    body = figure && <PlotView figure={figure} onPick={onPick} />;
  } else if (!hasResult) {
    body =
      stage.status === "running" ? (
        <div class="placeholder">
          {/* 0.05 is the placeholder set when a run starts; show a percentage only after real progress */}
          <MoleculeLoader caption={`Running ${stage.title}…`} progress={pct !== null && pct > 0.05 ? pct : null} />
        </div>
      ) : (
        <div class="placeholder">Set parameters and press Run</div>
      );
  } else if (tab === "table") {
    body = page && <DataTable page={page} />;
  } else {
    body = figure && <PlotView figure={figure} onPick={onPick} />;
  }

  return (
    <div class="layout">
      <JobBar pct={pct} />
      <Rail stages={stages} current={current} onSelect={select} onOpen={openProject} onSave={saveProject} onStyle={openStyle} />
      <main class="center">
        <div class="tabs" role="tablist">
          {tabs.length > 0 &&
            tabs.map((t) => (
              <button key={t} role="tab" aria-selected={t === tab} onClick={() => {
                setShowGuide(false);
                setTab(t);
              }}>
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
            {current === "import" && hasResult && (
              <button class="link" onClick={() => setShowGuide((v) => !v)}>
                {showGuide ? "Hide format" : "File format"}
              </button>
            )}
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
      {sideMode === "style" && globalStyle ? (
        <div class="panel">
          <h2>Plot style</h2>
          <p class="hint">Applies to every plot and export. Stages can override it in their Style section.</p>
          <StyleForm effective={globalStyle} onChange={patchGlobal} />
          <button type="button" class="btn" onClick={() => setSideMode("params")}>
            Done
          </button>
        </div>
      ) : (
      <ParamPanel
        key={`${current}:${formKey}`}
        title={stage.title}
        schema={stage.schema}
        initial={stage.params}
        errors={errors}
        busy={pct !== null}
        onRun={run}
        picked={picked}
        onValuesChange={current === "import" ? setImportValues : undefined}
        injected={injected}
      >
        {current === "stocsy" && (
          <StocsyTools
            recent={recent}
            onUse={(a) => {
              setInjected({ values: { ...a }, n: Date.now() });
              run({ ...a });
            }}
            onClear={() => setInjected({ values: { anchor_ppm: "" }, n: Date.now() })}
          />
        )}
        {current === "annotate" && (
          <AnnotationList
            items={annotations}
            onDelete={deleteAnnotation}
            onImport={importAnnotations}
            onExport={exportAnnotations}
          />
        )}
        {stageStyle && (
          <details class="style-override">
            <summary>Style{Object.keys(stageStyle.override).length ? " (custom)" : ""}</summary>
            <StyleForm
              effective={stageStyle.effective}
              override={stageStyle.override}
              onChange={patchStage}
              onReset={resetStage}
            />
          </details>
        )}
      </ParamPanel>
      )}
      {pending && (
        <AnnotationPopover ppm={pending.ppm} x={pending.x} y={pending.y} onAdd={addAnnotation} onCancel={() => setPending(null)} />
      )}
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
