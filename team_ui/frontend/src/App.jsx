import { useState } from 'react'
import Header from './components/Header.jsx'
import WebcamView from './components/WebcamView.jsx'
import StatusBadge from './components/StatusBadge.jsx'
import PredictionCard from './components/PredictionCard.jsx'
import HistoryList from './components/HistoryList.jsx'
import Controls from './components/Controls.jsx'
import ErrorBanner from './components/ErrorBanner.jsx'
import { MOCK_PREDICTION, MOCK_HISTORY } from './mockData.js'

// State shape follows DESIGN.md "State (in App)".
// Week 1 - static layout. Week 2 - real webcam (WebcamView owns the stream).
export default function App() {
  const [cameraOn, setCameraOn] = useState(false)
  const [status, setStatus] = useState('idle') // idle | not_talking | recording | processing
  const [prediction, setPrediction] = useState(MOCK_PREDICTION)
  const [history, setHistory] = useState(MOCK_HISTORY)
  const [error, setError] = useState(null)

  function handleStart() {
    setError(null)
    setCameraOn(true) // WebcamView requests getUserMedia; onReady/onError below finish the transition
  }

  function handleCameraReady() {
    setStatus('not_talking')
  }

  function handleCameraError(message) {
    setCameraOn(false)
    setStatus('idle')
    setError(message)
  }

  function handleStop() {
    setCameraOn(false)
    setStatus('idle')
  }

  // Week 3: lip crop + opening ratio arrive here each frame (or null when no face).
  // Nothing consumes it yet - Week 4 adds speaking detection + the 22-frame buffer.
  function handleLipData(_data) {}

  function handleClear() {
    setHistory([])
    setPrediction(null)
  }

  return (
    <div className="mx-auto flex min-h-full max-w-5xl flex-col">
      <Header />

      <main className="flex flex-1 flex-col gap-4 p-5 lg:flex-row">
        {/* Left column - webcam + controls */}
        <section className="flex flex-col gap-3 lg:w-3/5">
          <WebcamView
            cameraOn={cameraOn}
            onReady={handleCameraReady}
            onError={handleCameraError}
            onLipData={handleLipData}
          />
          <Controls
            cameraOn={cameraOn}
            onStart={handleStart}
            onStop={handleStop}
            onClear={handleClear}
          />
          <ErrorBanner message={error} onDismiss={() => setError(null)} />
        </section>

        {/* Right column - status, prediction, history */}
        <aside className="flex flex-col gap-4 lg:w-2/5">
          <div className="card p-4">
            <StatusBadge status={status} />
          </div>
          <div className="card p-4">
            <PredictionCard prediction={prediction} />
          </div>
          <div className="card p-4">
            <HistoryList history={history} />
          </div>
        </aside>
      </main>

      <footer className="border-t border-line px-5 py-3 text-xs text-slate-600">
        Week 3 · live webcam + lip detection · backend not connected yet
      </footer>
    </div>
  )
}
