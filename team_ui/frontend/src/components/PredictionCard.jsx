import { CONFIDENCE_THRESHOLD } from '../constants.js'

// Big predicted word + confidence bar. Below the threshold we show
// "Prediction uncertain" and keep the top guess small (DESIGN.md rule).
export default function PredictionCard({ prediction, predicting = false, vocab = [] }) {
  const pct = prediction ? Math.round(prediction.confidence * 100) : 0
  const uncertain = prediction && prediction.confidence < CONFIDENCE_THRESHOLD

  // probs (trained model only, not the stub) lines up index-for-index with vocab -
  // ranking them shows how close the model was to a different similar-sounding word,
  // which is the actual thing this project is trying to demonstrate.
  const ranked =
    prediction?.probs && prediction.probs.length === vocab.length
      ? vocab
          .map((word, i) => ({ word, p: prediction.probs[i] }))
          .sort((a, b) => b.p - a.p)
          .slice(0, 3)
      : null

  return (
    <div>
      <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-500">
        Prediction
      </p>

      {predicting ? (
        <p className="animate-pulse text-2xl font-semibold text-slate-400">predicting…</p>
      ) : !prediction ? (
        <p className="text-2xl font-semibold text-slate-600">—</p>
      ) : uncertain ? (
        <div>
          <p className="text-2xl font-semibold text-slate-400">Prediction uncertain</p>
          <p className="mt-1 text-sm text-slate-500">
            top guess: <span className="text-slate-300">{prediction.word}</span>
          </p>
        </div>
      ) : (
        <p className="text-5xl font-bold uppercase tracking-tight text-white">
          {prediction.word}
        </p>
      )}

      <div className="mt-4">
        <div className="mb-1 flex justify-between text-xs text-slate-500">
          <span>Confidence</span>
          <span>{pct}%</span>
        </div>
        <div className="h-2 w-full overflow-hidden rounded-full bg-base">
          <div
            className={`h-full rounded-full transition-all ${
              uncertain ? 'bg-slate-500' : 'bg-accent'
            }`}
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

      {ranked && !predicting && (
        <div className="mt-4">
          <p className="mb-1 text-xs font-semibold uppercase tracking-wider text-slate-500">
            Top matches
          </p>
          <ul className="space-y-1.5">
            {ranked.map(({ word, p }) => (
              <li key={word} className="flex items-center gap-2 text-xs">
                <span className="w-10 shrink-0 uppercase text-slate-400">{word}</span>
                <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-base">
                  <div
                    className="h-full rounded-full bg-accent/60"
                    style={{ width: `${Math.round(p * 100)}%` }}
                  />
                </div>
                <span className="w-9 shrink-0 text-right text-slate-500">
                  {Math.round(p * 100)}%
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
