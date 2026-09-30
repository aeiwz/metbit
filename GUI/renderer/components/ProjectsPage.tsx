import { useState } from "preact/hooks";
import { fmtDate, fmtSize } from "./PageNav";

interface Props {
  projects: ProjectEntry[];
  api: ProjectsApi;
  onOpen: (path: string) => void;
  onOpenFile: () => void;
  onChanged: () => void; // reload the list after a change
}

type Mode = { path: string; kind: "rename" | "trash" } | null;

export function ProjectsPage({ projects, api, onOpen, onOpenFile, onChanged }: Props) {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<Mode>(null);
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const q = query.trim().toLowerCase();
  const shown = projects.filter((p) => !q || p.name.toLowerCase().includes(q) || p.folder.toLowerCase().includes(q));

  // every action reports its error on the page and reloads the list
  const act = async (fn: () => Promise<unknown>) => {
    setError(null);
    try {
      await fn();
      setMode(null);
    } catch (e) {
      setError((e as Error).message.replace(/^Error invoking remote method '[^']+': (Error: )?/, ""));
    }
    onChanged();
  };

  return (
    <div class="projects">
      <div class="section-head">
        <h1>Projects</h1>
        <div class="row">
          <input type="search" class="search" placeholder="Search projects" aria-label="Search projects" value={query} onInput={(e) => setQuery(e.currentTarget.value)} />
          <button type="button" class="btn" onClick={onOpenFile}>
            Open project…
          </button>
        </div>
      </div>
      {error && <p class="error">{error}</p>}
      {projects.length === 0 ? (
        <p class="hint">No projects yet. Save an analysis with Save project… and it appears here.</p>
      ) : shown.length === 0 ? (
        <p class="hint">No project matches "{query}".</p>
      ) : (
        <table class="project-table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Modified</th>
              <th>Size</th>
              <th>
                <span class="sr-only">Actions</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {shown.map((p) => (
              <tr key={p.path} class={p.exists ? undefined : "missing"}>
                <td>
                  {mode?.path === p.path && mode.kind === "rename" ? (
                    <form
                      class="row"
                      onSubmit={(e) => {
                        e.preventDefault();
                        act(() => api.rename(p.path, name));
                      }}
                    >
                      <input aria-label="New name" value={name} onInput={(e) => setName(e.currentTarget.value)} autoFocus />
                      <button type="submit" class="btn primary">
                        Rename
                      </button>
                      <button type="button" class="link" onClick={() => setMode(null)}>
                        Cancel
                      </button>
                    </form>
                  ) : (
                    <>
                      <div class="project-name">{p.name}</div>
                      <div class="hint" title={p.path}>
                        {p.exists ? p.folder : `File not found: ${p.folder}`}
                      </div>
                    </>
                  )}
                </td>
                <td class="hint nowrap">{fmtDate(p.modifiedAt)}</td>
                <td class="hint nowrap">{fmtSize(p.size)}</td>
                <td>
                  <div class="project-actions">
                  {mode?.path === p.path && mode.kind === "trash" ? (
                    <span class="row confirm">
                      Move to Trash?
                      <button type="button" class="link danger" onClick={() => act(() => api.trash(p.path))}>
                        Yes
                      </button>
                      <button type="button" class="link" onClick={() => setMode(null)}>
                        No
                      </button>
                    </span>
                  ) : (
                    <>
                      <button type="button" class="link" disabled={!p.exists} onClick={() => onOpen(p.path)}>
                        Open
                      </button>
                      <button
                        type="button"
                        class="link"
                        disabled={!p.exists}
                        onClick={() => {
                          setName(p.name);
                          setMode({ path: p.path, kind: "rename" });
                        }}
                      >
                        Rename
                      </button>
                      <button type="button" class="link" disabled={!p.exists} onClick={() => act(() => api.duplicate(p.path))}>
                        Duplicate
                      </button>
                      <button type="button" class="link" disabled={!p.exists} onClick={() => act(() => api.reveal(p.path))}>
                        Show in folder
                      </button>
                      <button type="button" class="link" onClick={() => act(() => api.remove(p.path))}>
                        Remove from list
                      </button>
                      <button type="button" class="link danger" disabled={!p.exists} onClick={() => setMode({ path: p.path, kind: "trash" })}>
                        Move to Trash
                      </button>
                    </>
                  )}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
