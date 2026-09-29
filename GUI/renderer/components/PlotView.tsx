import Plotly from "plotly.js-dist-min";
import { useEffect, useRef } from "preact/hooks";

interface Figure {
  data: unknown[];
  layout: Record<string, any>;
}

export interface PickPoint {
  x: unknown;
  customdata: unknown;
  clientX?: number; // viewport position of the click, for popovers
  clientY?: number;
}

const BLACK = new Set(["black", "#000", "#000000", "rgb(0,0,0)", "rgb(0, 0, 0)"]);

// metbit figures use plotly_white with hard-coded black lines and labels; recolor them from the CSS tokens
// ponytail: read once per render, so an OS theme switch applies on the next figure load
function themed(layout: Record<string, any>): Record<string, any> {
  const css = getComputedStyle(document.documentElement);
  const text = css.getPropertyValue("--text").trim();
  const border = css.getPropertyValue("--border").trim();
  const out: Record<string, any> = {
    ...layout,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    // size comes from the plot style on the server
    font: { ...layout.font, family: css.getPropertyValue("--font"), color: text },
  };
  for (const key of Object.keys(layout)) {
    if (/^[xy]axis\d*$/.test(key)) {
      out[key] = {
        ...layout[key],
        automargin: true, // long tick labels (heatmap group names) get room instead of clipping
        gridcolor: border,
        linecolor: text,
        zerolinecolor: border,
        tickfont: { ...layout[key].tickfont, color: text },
      };
    }
  }
  if (layout.title) out.title = { ...layout.title, font: { ...layout.title.font, color: text } };
  if (layout.annotations) {
    out.annotations = layout.annotations.map((a: any) => ({ ...a, arrowcolor: text, font: { ...a.font, color: text } }));
  }
  if (layout.shapes) {
    out.shapes = layout.shapes.map((s: any) => (BLACK.has(String(s.line?.color).toLowerCase()) ? { ...s, line: { ...s.line, color: text } } : s));
  }
  return out;
}

export function PlotView({ figure, onPick }: { figure: Figure; onPick?: (point: PickPoint) => void }) {
  const el = useRef<HTMLDivElement>(null);
  const pick = useRef(onPick);
  pick.current = onPick; // latest callback without re-binding plotly listeners
  const pickable = onPick !== undefined;

  useEffect(() => {
    const node = el.current!;
    return () => Plotly.purge(node);
  }, []);

  useEffect(() => {
    const node: any = el.current!;
    const layout = themed(figure.layout);
    if (pickable) layout.hoverdistance = -1; // any click in the plot picks the nearest point, not only within 20 px
    // toImage removed: the tab bar's PNG / SVG export saves the original white-background figure instead
    const config = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["toImage"] };
    Plotly.react(node, figure.data, layout, config).then(() => {
      node.removeAllListeners?.("plotly_click");
      node.on("plotly_click", (e: { points?: PickPoint[]; event?: MouseEvent }) => {
        const p = e.points?.[0];
        if (p) pick.current?.({ x: p.x, customdata: p.customdata, clientX: e.event?.clientX, clientY: e.event?.clientY });
      });
    });
  }, [figure, pickable]);

  return <div ref={el} class="plot" />;
}
