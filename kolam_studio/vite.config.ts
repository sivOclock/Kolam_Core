import { defineConfig } from 'vite'

// The brush spec is shared with the Python generator: src/kolam/spec.ts imports
// ../kolam_press/brushes.json, so the dev server may read the parent folder.
export default defineConfig({
  base: './',
  server: { fs: { allow: ['..'] } },
})
