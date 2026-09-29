import { defineConfig } from "vitest/config";
import preact from "@preact/preset-vite";

export default defineConfig({
  plugins: [preact()],
  test: {
    environment: "jsdom",
    include: ["renderer/**/*.test.{ts,tsx}", "electron/**/*.test.ts"],
  },
});
