// Shows which words the model knows, so a new user knows what to say.
// The list comes from the backend (GET /vocab), not hardcoded here.
export default function VocabHint({ vocab = [] }) {
  if (vocab.length === 0) return null

  return (
    <div>
      <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-500">
        Words you can say
      </p>
      <div className="flex flex-wrap gap-2">
        {vocab.map((word) => (
          <span
            key={word}
            className="rounded-md border border-line bg-base px-2 py-1 text-sm font-medium uppercase text-slate-300"
          >
            {word}
          </span>
        ))}
      </div>
    </div>
  )
}
