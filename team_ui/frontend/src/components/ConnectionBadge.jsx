// Week 6 - shows the backend WebSocket state + which predictor is active.
const CONN = {
  connecting: { label: 'connecting…', dot: 'bg-amber-400' },
  open: { label: 'connected', dot: 'bg-emerald-500' },
  closed: { label: 'offline', dot: 'bg-slate-500' },
  error: { label: 'unreachable', dot: 'bg-rose-500' },
}

export default function ConnectionBadge({ connState, modelKind }) {
  const c = CONN[connState] ?? CONN.closed
  return (
    <div className="flex items-center gap-2 text-xs text-slate-500">
      <span className={`h-2 w-2 rounded-full ${c.dot}`} />
      <span>backend {c.label}</span>
      {connState === 'open' && modelKind && (
        <span className="rounded bg-base px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-slate-400">
          {modelKind === 'stub' ? 'stub predictor' : 'trained model'}
        </span>
      )}
    </div>
  )
}
