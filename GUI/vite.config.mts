import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import preact from "@preact/preset-vite";

export default defineConfig({
  root: "renderer",
  base: "./", // loaded via file:// in Electron
  plugins: [preact()],
  build: { outDir: "../dist", emptyOutDir: true },
  // the About page shows the repository's LICENSE, one level above GUI/
  server: { fs: { allow: [fileURLToPath(new URL("..", import.meta.url))] } },
});
