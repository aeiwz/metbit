interface BackendInfo {
  port: number;
  token: string;
}

interface Window {
  metbit: {
    backendInfo(): Promise<BackendInfo>;
    restartBackend(): Promise<BackendInfo>;
    openFile(filters?: { name: string; extensions: string[] }[], directory?: boolean): Promise<string | null>;
    onBackendExit(cb: () => void): void;
    saveDialog(defaultName: string, filters?: { name: string; extensions: string[] }[]): Promise<string | null>;
    saveImage(defaultName: string, dataUrl: string): Promise<string | null>;
  };
}

declare module "plotly.js-dist-min";
