import { app, BrowserWindow, dialog, ipcMain } from "electron";
import { join } from "node:path";
import { startSidecar, type Backend, type SidecarCommand } from "./sidecar";

let backend: Backend | null = null;
let win: BrowserWindow | null = null;
const logDir = () => join(app.getPath("userData"), "logs");

function sidecarCommand(): SidecarCommand {
  if (app.isPackaged) {
    const exe = process.platform === "win32" ? "metbit-server.exe" : "metbit-server";
    return { cmd: join(process.resourcesPath, "metbit-server", exe), args: [] };
  }
  // dev: run the sidecar from source with a Python that has metbit's deps
  const serverDir = join(__dirname, "..", "server");
  return {
    cmd: process.env.METBIT_PYTHON ?? (process.platform === "win32" ? "python" : "python3"),
    args: ["-m", "metbit_server"],
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
ipcMain.handle("dialog:openFile", async (_e, filters?: Electron.FileFilter[]) => {
  const r = await dialog.showOpenDialog(win!, { properties: ["openFile"], filters });
  return r.canceled ? null : r.filePaths[0];
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
