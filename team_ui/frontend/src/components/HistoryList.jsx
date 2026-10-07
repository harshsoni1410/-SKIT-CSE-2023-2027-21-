import { CONFIDENCE_THRESHOLD } from '../constants.js'

// Last ~10 predictions: word, confidence, time.
export default function HistoryList({ history = [] }) {
  // average confidence of the predictions shown below (0 if there are none)
  const shown = history.slice(0, 10)
  const avg = shown.length
    ? Math.round((shown.reduce((sum, item) => sum + item.confidence, 0) / shown.length) * 100)
    : 0

  return (
    <div>
      <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-500">
        History
      </p>

      {history.length === 0 ? (
        <p className="text-sm text-slate-600">No predictions yet.</p>
      ) : (
        <ul className="divide-y divide-line">
          {history.slice(0, 10).map((item, i) => (
            <li key={i} className="flex items-center justify-between py-2 text-sm">
              <span
                className={
                  item.confidence < CONFIDENCE_THRESHOLD
                    ? 'text-slate-500'
                    : 'font-medium text-slate-200'
                }
              >
                {item.word}
              </span>
              <span className="text-slate-500">{Math.round(item.confidence * 100)}%</span>
              <span className="font-mono text-xs text-slate-600">{item.time}</span>
            </li>
          ))}
        </ul>
      )}

      {shown.length > 0 && (
        <p className="mt-2 text-xs text-slate-500">
          Average confidence: <span className="text-slate-300">{avg}%</span> over{' '}
          {shown.length} prediction{shown.length === 1 ? '' : 's'}
        </p>
      )}
    </div>
  )
}
