// Camera permission denied, backend disconnected, etc.
export default function ErrorBanner({ message, onDismiss }) {
  if (!message) return null
  return (
    <div className="flex items-start justify-between gap-3 rounded-lg border border-rose-500/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
      <span>{message}</span>
      {onDismiss && (
        <button
          onClick={onDismiss}
          className="shrink-0 text-rose-300 hover:text-rose-100"
          aria-label="Dismiss"
        >
          ✕
        </button>
      )}
    </div>
  )
}
