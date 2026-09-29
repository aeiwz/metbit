export function dataUrlToBuffer(url: string): Buffer {
  const comma = url.indexOf(",");
  if (!url.startsWith("data:") || comma < 0) throw new Error("Not a data URL");
  const header = url.slice(5, comma);
  const body = url.slice(comma + 1);
  return header.endsWith(";base64") ? Buffer.from(body, "base64") : Buffer.from(decodeURIComponent(body), "utf8");
}
