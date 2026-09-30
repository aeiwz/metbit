import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("metbit", {
  backendInfo: () => ipcRenderer.invoke("backend:info"),
  restartBackend: () => ipcRenderer.invoke("backend:restart"),
  openFile: (filters?: { name: string; extensions: string[] }[], directory?: boolean) =>
    ipcRenderer.invoke("dialog:openFile", filters, directory),
  saveDialog: (defaultName: string, filters?: { name: string; extensions: string[] }[]) =>
    ipcRenderer.invoke("dialog:save", defaultName, filters),
  saveImage: (defaultName: string, dataUrl: string) => ipcRenderer.invoke("file:saveImage", defaultName, dataUrl),
  appVersion: () => ipcRenderer.invoke("app:version"),
  setTheme: (pref: string) => ipcRenderer.invoke("theme:set", pref),
  projects: {
    list: () => ipcRenderer.invoke("projects:list"),
    add: (path: string) => ipcRenderer.invoke("projects:add", path),
    remove: (path: string) => ipcRenderer.invoke("projects:remove", path),
    rename: (path: string, name: string) => ipcRenderer.invoke("projects:rename", path, name),
    duplicate: (path: string) => ipcRenderer.invoke("projects:duplicate", path),
    reveal: (path: string) => ipcRenderer.invoke("projects:reveal", path),
    trash: (path: string) => ipcRenderer.invoke("projects:trash", path),
  },
  onBackendExit: (cb: () => void) => {
    ipcRenderer.on("backend:exit", () => cb());
  },
});
