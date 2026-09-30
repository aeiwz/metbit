import Plotly from "plotly.js-dist-min";
import { useEffect, useRef, useState } from "preact/hooks";
import { onThemeChange } from "../themePref";

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

export function PlotView({
  figure,
  onPick,
  onLabelMove,
  onLabelClick,
  uirevision,
}: {
  figure: Figure;
  onPick?: (point: PickPoint) => void;
  onLabelMove?: (id: string, ax: number, ay: number) => void;
  onLabelClick?: (id: string) => void;
  uirevision?: string; // same value across figure updates keeps the user's zoom
}) {
  const el = useRef<HTMLDivElement>(null);
  const pick = useRef(onPick);
  pick.current = onPick; // latest callback without re-binding plotly listeners
  const pickable = onPick !== undefined;
  const move = useRef(onLabelMove);
  move.current = onLabelMove;
  const draggable = onLabelMove !== undefined;
  // the user's zoom per uirevision; Plotly's own uirevision drops it on reversed (ppm) axes
  const zoom = useRef<{ rev?: string; ranges: Record<string, unknown[]> }>({ ranges: {} });
  const labelClick = useRef(onLabelClick);
  labelClick.current = onLabelClick;

  useEffect(() => {
    const node = el.current!;
    return () => Plotly.purge(node);
  }, []);

  // the plot's colours come from the CSS tokens: redraw when the theme changes
  const [themeTick, setThemeTick] = useState(0);
  useEffect(() => onThemeChange(() => setThemeTick((t) => t + 1)), []);

  useEffect(() => {
    const node: any = el.current!;
    const layout = themed(figure.layout);
    if (!uirevision || zoom.current.rev !== uirevision) zoom.current = { rev: uirevision, ranges: {} };
    for (const [axis, range] of Object.entries(zoom.current.ranges)) layout[axis] = { ...layout[axis], range, autorange: false };
    if (pickable) layout.hoverdistance = -1; // any click in the plot picks the nearest point, not only within 20 px
    // toImage removed: the tab bar's PNG / SVG export saves the original white-background figure instead
    const config = {
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ["toImage"],
      // drag a label off a crowded peak, as metbit's annotator (Plotly editable annotations)
      ...(draggable ? { edits: { annotationTail: true } } : {}),
    };
    Plotly.react(node, figure.data, layout, config).then(() => {
      node.removeAllListeners?.("plotly_click");
      node.on("plotly_click", (e: { points?: PickPoint[]; event?: MouseEvent }) => {
        const p = e.points?.[0];
        if (p) pick.current?.({ x: p.x, customdata: p.customdata, clientX: e.event?.clientX, clientY: e.event?.clientY });
      });
      node.removeAllListeners?.("plotly_clickannotation");
      node.on("plotly_clickannotation", (e: { annotation?: { name?: string } }) => {
        if (e.annotation?.name) labelClick.current?.(e.annotation.name);
      });
      node.removeAllListeners?.("plotly_relayout");
      node.on("plotly_relayout", (e: Record<string, unknown>) => {
        for (const key of Object.keys(e)) {
          if (uirevision) {
            // zooms report "xaxis.range" (an array) or "xaxis.range[0]" / "[1]", depending on the gesture
            const whole = /^([xy]axis\d*)\.range$/.exec(key);
            if (whole) zoom.current.ranges[whole[1]] = e[key] as unknown[];
            const zoomed = /^([xy]axis\d*)\.range\[0\]$/.exec(key);
            if (zoomed) zoom.current.ranges[zoomed[1]] = [e[key], e[`${zoomed[1]}.range[1]`]];
            const reset = /^([xy]axis\d*)\.autorange$/.exec(key);
            if (reset) delete zoom.current.ranges[reset[1]];
          }
          // a dragged label reports annotations[i].ax / .ay; our labels carry their id as the annotation name
          const m = /^annotations\[(\d+)\]\.ax$/.exec(key);
          const a = m ? node.layout.annotations?.[Number(m[1])] : undefined;
          if (a?.name) move.current?.(a.name, Math.round(a.ax), Math.round(a.ay));
        }
      });
    });
  }, [figure, pickable, draggable, uirevision, themeTick]);

  return <div ref={el} class="plot" />;
}
