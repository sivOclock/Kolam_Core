import { defineConfig } from 'vite'

// The brush spec is shared with kolam_gen and kolam_feed: src/kolam/model.ts imports the repo's
// brushes.json, so the dev server may read the parent folder.
// Pages: the photo reader (index.html, the main page), which runs the model in the browser from
// public/model (`python -m kolam_feed export`); how it works (how.html); the drawing studio (studio.html)
// and the encoding viewer (encodings.html).
export default defineConfig({
  base: './',
  server: { fs: { allow: ['..'] } },
  worker: { format: 'es' }, // the reader's model worker (onnxruntime-web is an ES module)
  build: { rollupOptions: { input: { main: 'index.html', how: 'how.html', studio: 'studio.html', encodings: 'encodings.html' } } },
})
