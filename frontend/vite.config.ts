import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Dev server proxies the API to the backend (doc 14 §7).
// The backend port is configurable via VITE_BACKEND_PORT (make dev BACKEND_PORT=…);
// WS paths are literal per doc 05 §8.
const backendPort = process.env.VITE_BACKEND_PORT ?? "8080";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api": `http://localhost:${backendPort}`,
      "/ws": { target: `ws://localhost:${backendPort}`, ws: true },
    },
  },
});
