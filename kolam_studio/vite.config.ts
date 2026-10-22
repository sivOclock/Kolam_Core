import { defineConfig } from 'vite'

// The brush spec is shared with kolam_gen and kolam_feed: src/kolam/model.ts imports the repo's
// brushes.json, so the dev server may read the parent folder.
// Three pages: the drawing studio (index.html), the encoding viewer (encodings.html) and the photo reader
// (predict.html), which runs the model in the browser from public/model (`python -m kolam_feed export`).
export default defineConfig({
  base: './',
  server: { fs: { allow: ['..'] } },
  build: { rollupOptions: { input: { main: 'index.html', encodings: 'encodings.html', predict: 'predict.html' } } },
})
