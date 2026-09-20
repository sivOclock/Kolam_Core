import './styles/main.css'
import paper from 'paper'
import { type GridConfig } from './modules/schema'
import { HTML, sizeCanvas, drawGrid } from './modules/canvas'
import { initDraw, resetDraw, renderSeqOnEnc } from './modules/draw'
import { decodeLive, decodeLoop, stopDecode } from './modules/decoder'
import { parseSeq } from './modules/seq-parser'

document.querySelector<HTMLDivElement>('#app')!.innerHTML = HTML

const encCanvas = document.getElementById('enc-canvas') as HTMLCanvasElement
const decCanvas = document.getElementById('dec-canvas') as HTMLCanvasElement
const seqEl = document.getElementById('seq') as HTMLTextAreaElement
const encScope = new paper.PaperScope()
const decScope = new paper.PaperScope()

let cfg: GridConfig = { rows: 3, cols: 3, spacing: 40 }
let initialized = false

function resizeCanvases() {
  sizeCanvas(encCanvas)
  sizeCanvas(decCanvas)
  encScope.view.viewSize = new encScope.Size(encCanvas.clientWidth, encCanvas.clientHeight)
  decScope.view.viewSize = new decScope.Size(decCanvas.clientWidth, decCanvas.clientHeight)
}

function redraw() {
  drawGrid(encScope, encCanvas, cfg, 'draw')
  drawGrid(decScope, decCanvas, cfg, 'decode')
  stopDecode()
  resetDraw(cfg)
  seqEl.value = ''
}

const ro = new ResizeObserver(() => {
  if (!initialized) {
    initialized = true
    encScope.setup(encCanvas)
    decScope.setup(decCanvas)
    initDraw(
      encScope, encCanvas, cfg,
      (seq) => decodeLive(decScope, decCanvas, cfg, seq, seqEl),
      (seq) => decodeLoop(decScope, decCanvas, cfg, seq, seqEl),
    )
  }
  resizeCanvases()
  redraw()
})
ro.observe(document.getElementById('boards')!)

document.getElementById('inp-rows')!.addEventListener('input', (e) => {
  const v = parseInt((e.target as HTMLInputElement).value)
  if (v > 0) { cfg = { ...cfg, rows: v }; redraw() }
})
document.getElementById('inp-cols')!.addEventListener('input', (e) => {
  const v = parseInt((e.target as HTMLInputElement).value)
  if (v > 0) { cfg = { ...cfg, cols: v }; redraw() }
})

seqEl.addEventListener('keydown', (e) => {
  if (e.key === '[' && !e.metaKey && !e.ctrlKey) {
    e.preventDefault()
    const val = seqEl.value
    const pos = seqEl.selectionStart ?? val.length
    const before = val.slice(0, pos).trimEnd()
    const after = val.slice(pos)
    const needsArrow = before && !before.endsWith('---') && !before.endsWith('\n')
    const insert = (needsArrow ? ' → ' : '') + '[,]'
    seqEl.value = before + insert + after
    // Place cursor between [ and ,
    const cursorPos = before.length + (needsArrow ? 4 : 1)
    seqEl.selectionStart = seqEl.selectionEnd = cursorPos
    seqEl.dispatchEvent(new Event('input'))
  }
})

seqEl.addEventListener('input', () => {
  const seq = parseSeq(seqEl.value)
  renderSeqOnEnc(seq)
  stopDecode()
  decodeLive(decScope, decCanvas, cfg, seq, seqEl, false)
})

seqEl.addEventListener('focus', () => stopDecode())
seqEl.addEventListener('blur', () => {
  const seq = parseSeq(seqEl.value)
  if (seq.length > 0) decodeLoop(decScope, decCanvas, cfg, seq, seqEl)
})

document.getElementById('btn-undo')!.addEventListener('click', () => { stopDecode(); resetDraw(cfg); seqEl.value = '' })

document.getElementById('btn-copy')!.addEventListener('click', () => {
  navigator.clipboard.writeText(seqEl.value)
  const btn = document.getElementById('btn-copy')!
  btn.textContent = 'Copied!'
  btn.classList.add('copied')
  setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('copied') }, 1500)
})
