import { defineConfig } from 'vite'

// The brush spec is shared with kolam_gen and kolam_feed: src/kolam/model.ts imports the repo's
// brushes.json, so the dev server may read the parent folder.
// Three pages: the drawing studio (index.html), the encoding viewer (encodings.html) and the photo reader
// (predict.html), whose /api calls go to `python -m kolam_feed serve` on port 8765.
export default defineConfig({
  base: './',
  server: { fs: { allow: ['..'] }, proxy: { '/api': 'http://127.0.0.1:8765' } },
  build: { rollupOptions: { input: { main: 'index.html', encodings: 'encodings.html', predict: 'predict.html' } } },
})
