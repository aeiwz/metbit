import { render } from "preact";
import { expect, it, vi } from "vitest";
import { AboutPage } from "./AboutPage";
import { HomePage } from "./HomePage";
import { PageShell } from "./PageNav";
import { ProjectsPage } from "./ProjectsPage";

const tick = () => new Promise((r) => setTimeout(r));
const button = (root: HTMLElement, text: string) =>
  [...root.querySelectorAll("button")].find((b) => b.textContent?.trim() === text) as HTMLButtonElement;

const entry = (name: string, exists = true): ProjectEntry => ({
  path: `/data/${name}.metbit`,
  name,
  folder: "/data",
  openedAt: "2026-09-30T08:00:00Z",
  exists,
  size: exists ? 2_400_000 : undefined,
  modifiedAt: exists ? "2026-09-30T08:00:00Z" : undefined,
});

it("landing: starts, opens and continues, with recent projects and the Kawa Technology credit", () => {
  const root = document.createElement("div");
  const h = { onNew: vi.fn(), onContinue: vi.fn(), onOpenFile: vi.fn(), onOpenPath: vi.fn(), onNavigate: vi.fn() };
  render(<HomePage version="0.1.0" recent={[entry("urine"), entry("gone", false)]} hasSession {...h} />, root);
  button(root, "New analysis").click();
  button(root, "Continue session").click();
  button(root, "Open project…").click();
  expect([h.onNew, h.onContinue, h.onOpenFile].every((f) => f.mock.calls.length === 1)).toBe(true);
  const recent = root.querySelectorAll<HTMLButtonElement>(".recent-open");
  recent[0].click();
  expect(h.onOpenPath).toHaveBeenCalledWith("/data/urine.metbit");
  expect(recent[1].disabled).toBe(true); // missing file
  expect(root.textContent).toContain("File not found");
  expect(root.querySelector(".page-foot")!.textContent).toContain("Developed by Kawa Technology");
  button(root, "About & License").click();
  expect(h.onNavigate).toHaveBeenCalledWith("about");
});

it("landing: no Continue without a session, and a hint without projects", () => {
  const root = document.createElement("div");
  const noop = () => {};
  render(<HomePage version="0.1.0" recent={[]} hasSession={false} onNew={noop} onContinue={noop} onOpenFile={noop} onOpenPath={noop} onNavigate={noop} />, root);
  expect(button(root, "Continue session")).toBeUndefined();
  expect(root.textContent).toContain("Projects you save or open appear here.");
});

function projectsApi(): ProjectsApi {
  return {
    list: vi.fn(),
    add: vi.fn(),
    remove: vi.fn().mockResolvedValue(undefined),
    rename: vi.fn().mockResolvedValue("/data/new.metbit"),
    duplicate: vi.fn().mockResolvedValue("/data/urine copy.metbit"),
    reveal: vi.fn().mockResolvedValue(undefined),
    trash: vi.fn().mockResolvedValue(undefined),
  };
}

it("projects: search, open, rename, duplicate, reveal, remove and trash with confirmation", async () => {
  const root = document.createElement("div");
  const api = projectsApi();
  const h = { onOpen: vi.fn(), onOpenFile: vi.fn(), onChanged: vi.fn() };
  render(<ProjectsPage projects={[entry("urine"), entry("plasma")]} api={api} {...h} />, root);
  expect(root.textContent).toContain("2.3 MB");
  const search = root.querySelector<HTMLInputElement>('[aria-label="Search projects"]')!;
  search.value = "plas";
  search.dispatchEvent(new Event("input"));
  await tick();
  expect(root.querySelectorAll("tbody tr")).toHaveLength(1);
  search.value = "";
  search.dispatchEvent(new Event("input"));
  await tick();

  button(root, "Open").click();
  expect(h.onOpen).toHaveBeenCalledWith("/data/urine.metbit");

  button(root, "Rename").click();
  await tick();
  const name = root.querySelector<HTMLInputElement>('[aria-label="New name"]')!;
  expect(name.value).toBe("urine");
  name.value = "urine 2026";
  name.dispatchEvent(new Event("input"));
  await tick();
  root.querySelector("form")!.dispatchEvent(new Event("submit", { cancelable: true }));
  await tick();
  expect(api.rename).toHaveBeenCalledWith("/data/urine.metbit", "urine 2026");

  button(root, "Duplicate").click();
  button(root, "Show in folder").click();
  button(root, "Remove from list").click();
  await tick();
  expect(api.duplicate).toHaveBeenCalledWith("/data/urine.metbit");
  expect(api.reveal).toHaveBeenCalledWith("/data/urine.metbit");
  expect(api.remove).toHaveBeenCalledWith("/data/urine.metbit");

  button(root, "Move to Trash").click();
  await tick();
  expect(api.trash).not.toHaveBeenCalled(); // asks first
  expect(root.textContent).toContain("Move to Trash?");
  button(root, "Yes").click();
  await tick();
  expect(api.trash).toHaveBeenCalledWith("/data/urine.metbit");
  expect(h.onChanged).toHaveBeenCalled();
});

it("projects: shows a failed action's message in plain words", async () => {
  const root = document.createElement("div");
  const api = projectsApi();
  api.duplicate = vi.fn().mockRejectedValue(new Error("Error invoking remote method 'projects:duplicate': Error: disk full"));
  render(<ProjectsPage projects={[entry("urine")]} api={api} onOpen={vi.fn()} onOpenFile={vi.fn()} onChanged={vi.fn()} />, root);
  button(root, "Duplicate").click();
  await vi.waitFor(() => expect(root.querySelector(".error")!.textContent).toBe("disk full"));
});

it("about: the MIT license verbatim, the Kawa Technology credit and open-source notices", () => {
  const root = document.createElement("div");
  render(<AboutPage version="0.1.0" />, root);
  const license = root.querySelector(".license")!.textContent!;
  expect(license.startsWith("MIT License")).toBe(true);
  expect(license).toContain("Copyright (c) 2024 Theerayut");
  expect(license).toContain('THE SOFTWARE IS PROVIDED "AS IS"');
  expect(root.querySelector(".kawa")!.textContent).toBe("Kawa Technology");
  expect(root.textContent).toContain("0.1.0");
  expect(root.querySelectorAll(".notices tbody tr").length).toBeGreaterThan(10);
});

it("page shell: navigates and marks the current page", () => {
  const root = document.createElement("div");
  const onNavigate = vi.fn();
  const onWorkspace = vi.fn();
  render(
    <PageShell page="projects" onNavigate={onNavigate} onWorkspace={onWorkspace}>
      <p>body</p>
    </PageShell>,
    root,
  );
  expect(button(root, "Projects").getAttribute("aria-current")).toBe("page");
  button(root, "About & License").click();
  button(root, "Workspace →").click();
  expect(onNavigate).toHaveBeenCalledWith("about");
  expect(onWorkspace).toHaveBeenCalled();
});
