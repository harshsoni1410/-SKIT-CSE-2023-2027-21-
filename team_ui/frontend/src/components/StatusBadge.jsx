// Maps the app status to a label + color. Status values match App state
// ("idle" | "not_talking" | "recording" | "processing") from DESIGN.md.
const STATUS_MAP = {
  idle: { label: 'CAMERA OFF', dot: 'bg-slate-500', text: 'text-slate-400' },
  not_talking: { label: 'NOT TALKING', dot: 'bg-slate-400', text: 'text-slate-300' },
  recording: { label: 'RECORDING', dot: 'bg-rose-500 animate-pulse', text: 'text-rose-300' },
  processing: { label: 'PROCESSING', dot: 'bg-amber-400 animate-pulse', text: 'text-amber-300' },
}

export default function StatusBadge({ status = 'idle' }) {
  const s = STATUS_MAP[status] ?? STATUS_MAP.idle
  return (
    <div>
      <p className="mb-1 text-xs font-semibold uppercase tracking-wider text-slate-500">
        Status
      </p>
      <div className="inline-flex items-center gap-2 rounded-lg border border-line bg-base px-3 py-1.5">
        <span className={`h-2 w-2 rounded-full ${s.dot}`} />
        <span className={`text-sm font-medium ${s.text}`}>{s.label}</span>
      </div>
    </div>
  )
}
