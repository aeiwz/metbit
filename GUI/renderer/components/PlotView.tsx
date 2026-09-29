import Plotly from "plotly.js-dist-min";
import { useEffect, useRef } from "preact/hooks";

interface Figure {
  data: unknown[];
  layout: Record<string, any>;
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
    font: { ...layout.font, family: css.getPropertyValue("--font"), size: 12, color: text },
  };
  for (const key of Object.keys(layout)) {
    if (/^[xy]axis\d*$/.test(key)) {
      out[key] = { ...layout[key], gridcolor: border, linecolor: text, zerolinecolor: border, tickfont: { ...layout[key].tickfont, color: text } };
    }
  }
  if (layout.title) out.title = { ...layout.title, font: { ...layout.title.font, color: text } };
  if (layout.annotations) out.annotations = layout.annotations.map((a: any) => ({ ...a, font: { ...a.font, color: text } }));
  if (layout.shapes) {
    out.shapes = layout.shapes.map((s: any) => (BLACK.has(String(s.line?.color).toLowerCase()) ? { ...s, line: { ...s.line, color: text } } : s));
  }
  return out;
}

export function PlotView({ figure }: { figure: Figure }) {
  const el = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const node = el.current!;
    return () => Plotly.purge(node);
  }, []);

  useEffect(() => {
    Plotly.react(el.current!, figure.data, themed(figure.layout), { responsive: true, displaylogo: false });
  }, [figure]);

  return <div ref={el} class="plot" />;
}
