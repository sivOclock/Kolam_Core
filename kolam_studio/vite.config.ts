import { defineConfig } from 'vite'

// The brush spec is shared with kolam_gen and kolam_feed: src/kolam/model.ts imports the repo's
// brushes.json, so the dev server may read the parent folder.
// Two pages: the drawing studio (index.html) and the encoding viewer (encodings.html).
export default defineConfig({
  base: './',
  server: { fs: { allow: ['..'] } },
  build: { rollupOptions: { input: { main: 'index.html', encodings: 'encodings.html' } } },
})
