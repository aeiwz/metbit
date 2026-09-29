interface BackendInfo {
  port: number;
  token: string;
}

interface Window {
  metbit: {
    backendInfo(): Promise<BackendInfo>;
    restartBackend(): Promise<BackendInfo>;
    openFile(filters?: { name: string; extensions: string[] }[]): Promise<string | null>;
    onBackendExit(cb: () => void): void;
  };
}

declare module "plotly.js-dist-min";
