import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("metbit", {
  backendInfo: () => ipcRenderer.invoke("backend:info"),
  restartBackend: () => ipcRenderer.invoke("backend:restart"),
  openFile: (filters?: { name: string; extensions: string[] }[], directory?: boolean) =>
    ipcRenderer.invoke("dialog:openFile", filters, directory),
  onBackendExit: (cb: () => void) => {
    ipcRenderer.on("backend:exit", () => cb());
  },
});
