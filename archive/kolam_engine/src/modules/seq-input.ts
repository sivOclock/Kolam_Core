// seq-input.ts
// Digits → fill row, Space/Comma → move to col, Space → commit token, Enter → commit + stroke break

import { parseSeq } from './seq-parser'

type OnChange = (seq: [number,number][][]) => void

const MARKER = '\u200B'

export function initSeqInput(seqEl: HTMLTextAreaElement, onChange: OnChange) {
  let row = '', col = ''
  let phase: 'row' | 'col' = 'row'
  let committed = ''

  function preview() {
    let hint = ''
    if (phase === 'row' && row) hint = `[${row},_]`
    else if (phase === 'col') hint = `[${row},${col || '_'}]`

    if (hint) {
      const base = committed.trimEnd()
      const sep = base && !base.endsWith('---') && !base.endsWith('\n') ? ' → ' : ''
      seqEl.value = committed + sep + MARKER + hint
    } else {
      seqEl.value = committed
    }
    seqEl.scrollTop = seqEl.scrollHeight
  }

  function commitToken(breakStroke: boolean) {
    if (row && col) {
      const token = `[${row},${col}]`
      const base = committed.trimEnd()
      committed = (!base || base.endsWith('---') || base.endsWith('\n'))
        ? (base ? base + '\n' : '') + token
        : base + ' → ' + token
      onChange(parseSeq(committed))
    }
    row = ''; col = ''; phase = 'row'
    if (breakStroke && committed.trim()) {
      committed = committed.trimEnd() + '\n---\n'
      onChange(parseSeq(committed))
    }
    preview()
  }

  seqEl.addEventListener('paste', (e) => {
    e.preventDefault()
    const text = e.clipboardData?.getData('text') ?? ''
    committed = text.trim()
    row = ''; col = ''; phase = 'row'
    seqEl.value = committed
    onChange(parseSeq(committed))
  })

  seqEl.addEventListener('keydown', (e) => {
    if (e.metaKey || e.ctrlKey) return // allow Cmd+C, Cmd+A etc

    if (e.key === 'Enter') {
      e.preventDefault()
      commitToken(true)
      return
    }

    if (e.key === ' ' || e.key === ',') {
      e.preventDefault()
      if (phase === 'row' && row) { phase = 'col'; preview() }
      else if (phase === 'col' && col) { commitToken(false) }
      return
    }

    if (e.key === 'Backspace') {
      e.preventDefault()
      if (phase === 'col' && col) col = col.slice(0, -1)
      else if (phase === 'col' && !col) { phase = 'row' }
      else if (phase === 'row' && row) row = row.slice(0, -1)
      else {
        // backspace into committed text
        committed = committed.slice(0, -1)
        onChange(parseSeq(committed))
      }
      preview()
      return
    }

    if (/^\d$/.test(e.key)) {
      e.preventDefault()
      if (phase === 'row') row += e.key
      else col += e.key
      preview()
      return
    }

    e.preventDefault()
  })
}
