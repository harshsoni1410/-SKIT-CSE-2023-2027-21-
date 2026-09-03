import { CONFIDENCE_THRESHOLD } from '../constants.js'

// Last ~10 predictions: word, confidence, time.
export default function HistoryList({ history = [] }) {
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
    </div>
  )
}
