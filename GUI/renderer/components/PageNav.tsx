import type { ComponentChildren } from "preact";
import { Lockup } from "./Brand";
import { ThemeToggle } from "./ThemeToggle";

export type Page = "home" | "projects" | "about";

const LINKS: { page: Page; label: string }[] = [
  { page: "home", label: "Home" },
  { page: "projects", label: "Projects" },
  { page: "about", label: "About & License" },
];

export const fmtDate = (iso?: string) =>
  iso ? new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }) : "";

export function fmtSize(bytes?: number): string {
  if (bytes === undefined) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

interface Props {
  page: Page;
  onNavigate: (page: Page) => void;
  onWorkspace: () => void;
  children: ComponentChildren;
}

// the frame of the landing, projects and about pages: a top bar, then the page
export function PageShell({ page, onNavigate, onWorkspace, children }: Props) {
  return (
    <div class="page">
      <header class="page-bar">
        <span class="rail-brand">
          <Lockup gui />
        </span>
        <nav aria-label="Pages">
          {LINKS.map((l) => (
            <button key={l.page} type="button" aria-current={l.page === page ? "page" : undefined} onClick={() => onNavigate(l.page)}>
              {l.label}
            </button>
          ))}
        </nav>
        <ThemeToggle />
        <button type="button" class="btn" onClick={onWorkspace}>
          Workspace →
        </button>
      </header>
      <main class="page-body">{children}</main>
    </div>
  );
}
