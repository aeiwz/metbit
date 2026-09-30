import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";
import preact from "@preact/preset-vite";

export default defineConfig({
  plugins: [preact()],
  server: { fs: { allow: [fileURLToPath(new URL("..", import.meta.url))] } }, // LICENSE for the About page
  test: {
    environment: "jsdom",
    include: ["renderer/**/*.test.{ts,tsx}", "electron/**/*.test.ts"],
  },
});
