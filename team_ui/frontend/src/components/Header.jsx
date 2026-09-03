export default function Header() {
  return (
    <header className="flex items-baseline justify-between border-b border-line px-5 py-4">
      <div className="flex items-baseline gap-3">
        <span className="text-xl font-bold tracking-tight text-white">LipSense</span>
        <span className="hidden text-sm text-slate-400 sm:inline">
          Real-Time AI Lip Reading (3D CNN)
        </span>
      </div>
      <span className="text-xs text-slate-500">SKIT · CSE · Group 21</span>
    </header>
  )
}
