// Encoding viewer: paste kolam encodings (spec §12), get one chalkboard per kolam.
// encoding -> fromEncoding -> compile -> drawn, the same path as the drawing studio's kolam board.

import './style.css'
import { Board, CHALK, bezierSegments } from './board'
import { compile } from './kolam/compiler'
import { drawn } from './kolam/drawn'
import { fromEncoding, parseEncodings } from './kolam/encoding'
import { formatProgram, validate } from './kolam/model'

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <div class="enc-page">
    <div class="enc-input">
      <div class="panel-top">
        <span class="title">kolam encodings</span>
        <button id="clear">clear</button>
      </div>
      <textarea id="encodings" spellcheck="false" aria-label="Kolam encodings"
        placeholder="Paste encodings: rows of digits, a blank line between kolams (data/encodings_100.txt).
Also one per line as a JSON list, a labels.jsonl record, or rows joined by |"></textarea>
      <div id="status" class="status" role="status"></div>
    </div>
    <div id="grid" class="enc-grid"></div>
  </div>`

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T
const input = $<HTMLTextAreaElement>('encodings')
const grid = $<HTMLDivElement>('grid')
const status = $<HTMLDivElement>('status')
let boards: { board: Board; draw: () => void }[] = []

function render(): void {
  const encs = parseEncodings(input.value)
  grid.innerHTML = ''
  boards = []
  let bad = 0
  encs.forEach((enc, n) => {
    const card = document.createElement('div')
    card.className = 'enc-card'
    const canvas = document.createElement('canvas')
    canvas.className = 'board'
    const note = document.createElement('div')
    note.className = 'status'
    const pre = document.createElement('pre')
    card.append(canvas, note, pre)
    grid.append(card)
    note.onclick = () => card.classList.toggle('open')
    try {
      const h = fromEncoding(enc)
      const k = compile(h)
      const problems = validate(h)
      pre.textContent = enc.join('\n') + '\n\n' + formatProgram(h)
      const count = (x: number, w: string) => `${x} ${w}${x === 1 ? '' : 's'}`
      note.textContent = `${h.rows}×${h.cols} · ${count(h.relations.length, 'relation')} · ${count(k.strokes.length, 'stroke')}`
      if (problems.length) {
        note.textContent += ` · ${problems[0].message}`
        note.classList.add('error')
      }
      const board = new Board(canvas, `${n + 1}`)
      const draw = () => {
        board.layout(h.rows, h.cols, 1) // room for crescents pointing off the grid
        board.clear()
        board.dots(CHALK.main, Math.max(2, board.s * 0.14), k.used) // unused dots are not drawn
        const w = Math.max(1.5, board.s * 0.14)
        drawn(k).strokes.forEach((P, si) =>
          bezierSegments(board, P).forEach((seg, i) => board.chalk(seg, CHALK.main, w, si * 1009 + i * 7 + 1)),
        )
      }
      boards.push({ board, draw })
      draw()
    } catch (e) {
      bad++
      pre.textContent = enc.join('\n')
      note.textContent = `${n + 1}: ${(e as Error).message}`
      note.classList.add('error')
      card.classList.add('open')
    }
  })
  status.classList.toggle('error', bad > 0)
  status.textContent = encs.length
    ? `${encs.length} kolams${bad ? ` · ${bad} could not be read` : ''} · click a caption for its digits and program`
    : ''
}

let timer = 0
input.addEventListener('input', () => {
  clearTimeout(timer)
  timer = window.setTimeout(render, 250)
})
$<HTMLButtonElement>('clear').onclick = () => {
  input.value = ''
  render()
}
let resize = 0
window.addEventListener('resize', () => {
  clearTimeout(resize)
  resize = window.setTimeout(() => boards.forEach(({ board, draw }) => (board.invalidateBase(), draw())), 150)
})
document.fonts.ready.then(() => boards.forEach(({ board, draw }) => (board.invalidateBase(), draw())))
