import { CONFIDENCE_THRESHOLD } from '../constants.js'

// Prediction history -> CSV text, oldest first (the UI list is newest first).
// Columns: time, word, confidence (0..1), uncertain (true when below the
// threshold the UI greys out).
export function historyToCsv(history) {
  const rows = [['time', 'word', 'confidence', 'uncertain']]
  for (const item of [...history].reverse()) {
    rows.push([
      item.time,
      item.word,
      item.confidence.toFixed(3),
      String(item.confidence < CONFIDENCE_THRESHOLD),
    ])
  }
  return rows.map((r) => r.map(csvCell).join(',')).join('\r\n') + '\r\n'
}

// Quote a cell only when it needs it (comma, quote or newline inside).
function csvCell(value) {
  const s = String(value)
  return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
}

// Ask the browser to save the history as lipsense-history-YYYY-MM-DD.csv.
export function downloadHistoryCsv(history) {
  const blob = new Blob([historyToCsv(history)], { type: 'text/csv;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `lipsense-history-${new Date().toISOString().slice(0, 10)}.csv`
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}
