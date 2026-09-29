import { render } from "preact";

async function probe(): Promise<string> {
  const { port, token } = await window.metbit.backendInfo();
  const r = await fetch(`http://127.0.0.1:${port}/health`, { headers: { Authorization: `Bearer ${token}` } });
  return r.ok ? `Backend ready on port ${port}` : `Backend error ${r.status}`;
}

probe().then((msg) => render(<p>{msg}</p>, document.getElementById("app")!));
