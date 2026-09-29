import { spawn, type ChildProcess } from "node:child_process";
import { randomBytes } from "node:crypto";
import { createWriteStream, existsSync, mkdirSync, renameSync, statSync, type WriteStream } from "node:fs";
import { createServer } from "node:net";
import { join } from "node:path";

export interface SidecarCommand {
  cmd: string;
  args: string[];
  cwd?: string;
  env?: Record<string, string>;
}

export interface Backend {
  port: number;
  token: string;
  proc: ChildProcess;
  logFile: string;
}

export function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const srv = createServer();
    srv.once("error", reject);
    srv.listen(0, "127.0.0.1", () => {
      const { port } = srv.address() as { port: number };
      srv.close(() => resolve(port));
    });
  });
}

export async function waitForHealth(url: string, token: string, timeoutMs: number): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const r = await fetch(`${url}/health`, { headers: { Authorization: `Bearer ${token}` } });
      if (r.ok) return;
    } catch {
      // not listening yet
    }
    await new Promise((r) => setTimeout(r, 200));
  }
  throw new Error(`Backend not ready after ${timeoutMs} ms`);
}

const MAX_LOG_BYTES = 5 * 1024 * 1024;

export function openLog(dir: string): { file: string; stream: WriteStream } {
  mkdirSync(dir, { recursive: true });
  const file = join(dir, "server.log");
  // ponytail: one backup file; add numbered rotation if anyone needs older logs
  if (existsSync(file) && statSync(file).size > MAX_LOG_BYTES) renameSync(file, `${file}.1`);
  return { file, stream: createWriteStream(file, { flags: "a" }) };
}

export async function startSidecar(command: SidecarCommand, logDir: string): Promise<Backend> {
  const port = await freePort();
  const token = randomBytes(24).toString("hex");
  const log = openLog(logDir);
  const proc = spawn(command.cmd, [...command.args, "--port", String(port)], {
    cwd: command.cwd,
    env: { ...process.env, ...command.env, METBIT_TOKEN: token },
  });
  proc.stdout?.pipe(log.stream);
  proc.stderr?.pipe(log.stream);
  try {
    await waitForHealth(`http://127.0.0.1:${port}`, token, 20_000);
  } catch (e) {
    proc.kill();
    throw e;
  }
  return { port, token, proc, logFile: log.file };
}
