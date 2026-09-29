// @vitest-environment node
import { createServer } from "node:http";
import { afterEach, expect, it } from "vitest";
import { freePort, waitForHealth } from "./sidecar";

let close: (() => void) | undefined;
afterEach(() => close?.());

it("resolves once /health answers 200 with the right token", async () => {
  const port = await freePort();
  const srv = createServer((req, res) => {
    const ok = req.url === "/health" && req.headers.authorization === "Bearer t";
    res.writeHead(ok ? 200 : 401).end();
  });
  await new Promise<void>((r) => srv.listen(port, "127.0.0.1", r));
  close = () => srv.close();
  await expect(waitForHealth(`http://127.0.0.1:${port}`, "t", 2000)).resolves.toBeUndefined();
});

it("rejects after the timeout when nothing listens", async () => {
  const port = await freePort();
  await expect(waitForHealth(`http://127.0.0.1:${port}`, "t", 500)).rejects.toThrow("Backend not ready after 500 ms");
});
