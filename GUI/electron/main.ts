import { app, BrowserWindow, dialog, ipcMain } from "electron";
import { writeFile } from "node:fs/promises";
import { extname, join } from "node:path";
import { dataUrlToBuffer } from "./files";
import { startSidecar, type Backend, type SidecarCommand } from "./sidecar";

// test seams: Playwright cannot drive native dialogs, and tests keep their own userData
const E2E_DIR = process.env.METBIT_E2E_DIR;
const E2E_OPEN = process.env.METBIT_E2E_OPEN;
if (process.env.METBIT_USER_DATA) app.setPath("userData", process.env.METBIT_USER_DATA);

let backend: Backend | null = null;
let win: BrowserWindow | null = null;
const logDir = () => join(app.getPath("userData"), "logs");

const autosavePath = () => join(app.getPath("userData"), "autosave.metbit");

function sidecarCommand(): SidecarCommand {
  if (app.isPackaged) {
    const exe = process.platform === "win32" ? "metbit-server.exe" : "metbit-server";
    return { cmd: join(process.resourcesPath, "metbit-server", exe), args: ["--autosave", autosavePath()] };
  }
  // dev: run the sidecar from source with a Python that has metbit's deps
  const serverDir = join(__dirname, "..", "server");
  return {
    cmd: process.env.METBIT_PYTHON ?? (process.platform === "win32" ? "python" : "python3"),
    args: ["-m", "metbit_server", "--autosave", autosavePath()],
    cwd: serverDir,
    env: { PYTHONPATH: join(serverDir, "..", "..") },
  };
}

async function launchBackend(): Promise<Backend> {
  const b = await startSidecar(sidecarCommand(), logDir());
  b.proc.on("exit", () => {
    if (backend === b) win?.webContents.send("backend:exit"); // ignore deliberate restarts
  });
  backend = b;
  return b;
}

const info = (b: Backend) => ({ port: b.port, token: b.token });

ipcMain.handle("backend:info", () => (backend ? info(backend) : null));
ipcMain.handle("backend:restart", async () => {
  const old = backend;
  backend = null;
  old?.proc.kill();
  return info(await launchBackend());
});
ipcMain.handle("dialog:openFile", async (_e, filters?: Electron.FileFilter[], directory?: boolean) => {
  if (E2E_OPEN) return E2E_OPEN;
  const r = await dialog.showOpenDialog(win!, { properties: [directory ? "openDirectory" : "openFile"], filters });
  return r.canceled ? null : r.filePaths[0];
});

async function chooseSavePath(defaultName: string, filters?: Electron.FileFilter[]): Promise<string | null> {
  if (E2E_DIR) return join(E2E_DIR, defaultName);
  const r = await dialog.showSaveDialog(win!, { defaultPath: defaultName, filters });
  return r.canceled || !r.filePath ? null : r.filePath;
}

ipcMain.handle("dialog:save", (_e, defaultName: string, filters?: Electron.FileFilter[]) => chooseSavePath(defaultName, filters));
ipcMain.handle("file:saveImage", async (_e, defaultName: string, dataUrl: string) => {
  const path = await chooseSavePath(defaultName, [{ name: "Image", extensions: [extname(defaultName).slice(1)] }]);
  if (path) await writeFile(path, dataUrlToBuffer(dataUrl));
  return path;
});

async function boot() {
  try {
    await launchBackend();
  } catch (e) {
    dialog.showErrorBox("metbit could not start", `${(e as Error).message}\n\nLog: ${join(logDir(), "server.log")}`);
    app.quit();
    return;
  }
  win = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 960,
    minHeight: 600,
    title: "metbit",
    webPreferences: {
      preload: join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  win.loadFile(join(__dirname, "..", "dist", "index.html"));
}

app.whenReady().then(boot);
app.on("window-all-closed", () => app.quit());
app.on("will-quit", () => backend?.proc.kill());
