import { defineConfig } from "vite";
import preact from "@preact/preset-vite";

export default defineConfig({
  root: "renderer",
  base: "./", // loaded via file:// in Electron
  plugins: [preact()],
  build: { outDir: "../dist", emptyOutDir: true },
});
