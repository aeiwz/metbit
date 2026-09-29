import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("metbit", {
  backendInfo: () => ipcRenderer.invoke("backend:info"),
  restartBackend: () => ipcRenderer.invoke("backend:restart"),
  openFile: (filters?: { name: string; extensions: string[] }[], directory?: boolean) =>
    ipcRenderer.invoke("dialog:openFile", filters, directory),
  saveDialog: (defaultName: string, filters?: { name: string; extensions: string[] }[]) =>
    ipcRenderer.invoke("dialog:save", defaultName, filters),
  saveImage: (defaultName: string, dataUrl: string) => ipcRenderer.invoke("file:saveImage", defaultName, dataUrl),
  onBackendExit: (cb: () => void) => {
    ipcRenderer.on("backend:exit", () => cb());
  },
});
