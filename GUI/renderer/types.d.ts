interface BackendInfo {
  port: number;
  token: string;
}

interface ProjectEntry {
  path: string;
  name: string;
  folder: string;
  openedAt: string;
  exists: boolean;
  size?: number;
  modifiedAt?: string;
}

interface ProjectsApi {
  list(): Promise<ProjectEntry[]>;
  add(path: string): Promise<void>;
  remove(path: string): Promise<void>;
  rename(path: string, name: string): Promise<string>;
  duplicate(path: string): Promise<string>;
  reveal(path: string): Promise<void>;
  trash(path: string): Promise<void>;
}

declare module "*?raw" {
  const text: string;
  export default text;
}

interface Window {
  metbit: {
    backendInfo(): Promise<BackendInfo>;
    restartBackend(): Promise<BackendInfo>;
    openFile(filters?: { name: string; extensions: string[] }[], directory?: boolean): Promise<string | null>;
    onBackendExit(cb: () => void): void;
    saveDialog(defaultName: string, filters?: { name: string; extensions: string[] }[]): Promise<string | null>;
    saveImage(defaultName: string, dataUrl: string): Promise<string | null>;
    appVersion(): Promise<string>;
    projects: ProjectsApi;
  };
}

declare module "plotly.js-dist-min";
