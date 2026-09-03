export default function Controls({ cameraOn, onStart, onStop, onClear }) {
  return (
    <div className="flex flex-wrap gap-2">
      <button
        onClick={onStart}
        disabled={cameraOn}
        className="rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-base transition hover:bg-accent-soft disabled:cursor-not-allowed disabled:opacity-40"
      >
        Start Camera
      </button>
      <button
        onClick={onStop}
        disabled={!cameraOn}
        className="rounded-lg border border-line px-4 py-2 text-sm font-medium text-slate-300 transition hover:bg-card disabled:cursor-not-allowed disabled:opacity-40"
      >
        Stop
      </button>
      <button
        onClick={onClear}
        className="rounded-lg border border-line px-4 py-2 text-sm font-medium text-slate-300 transition hover:bg-card"
      >
        Clear History
      </button>
    </div>
  )
}
