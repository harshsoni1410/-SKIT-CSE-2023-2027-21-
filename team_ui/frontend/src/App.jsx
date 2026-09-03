import { useCallback, useRef, useState } from 'react'
import Header from './components/Header.jsx'
import WebcamView from './components/WebcamView.jsx'
import StatusBadge from './components/StatusBadge.jsx'
import PredictionCard from './components/PredictionCard.jsx'
import HistoryList from './components/HistoryList.jsx'
import Controls from './components/Controls.jsx'
import ErrorBanner from './components/ErrorBanner.jsx'
import { useSpeechCapture } from './hooks/useSpeechCapture.js'
import { sequenceToTensor } from './lib/speechCapture.js'
import { MOCK_PREDICTION, MOCK_HISTORY } from './mockData.js'

// State shape follows DESIGN.md "State (in App)".
//  Week 1 static · Week 2 webcam · Week 3 lip detection
//  Week 4: speaking detection + 22-frame sequence buffer (no backend yet -
//          a captured utterance is logged; real prediction arrives in Week 6)
export default function App() {
  const [cameraOn, setCameraOn] = useState(false)
  const [prediction, setPrediction] = useState(MOCK_PREDICTION)
  const [history, setHistory] = useState(MOCK_HISTORY)
  const [error, setError] = useState(null)
  const [notice, setNotice] = useState(null)
  const lastSequenceRef = useRef(null)

  const handleUtterance = useCallback((sequence) => {
    // sequence = 22 lip-crop canvases (80x112). Week 5/6 sends this to the backend.
    lastSequenceRef.current = sequence
    const tensor = sequenceToTensor(sequence)
    console.log('captured utterance', tensor.shape, tensor.data.length, 'floats')
    setNotice(`Captured a ${sequence.length}-frame sequence — prediction is wired up in Week 6.`)
    setTimeout(() => {
      setNotice(null)
      speech.ready()
    }, 1200)
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const speech = useSpeechCapture({ onUtterance: handleUtterance })

  const status = cameraOn ? speech.status : 'idle'

  function handleStart() {
    setError(null)
    setCameraOn(true)
  }

  function handleCameraError(message) {
    setCameraOn(false)
    speech.reset()
    setError(message)
  }

  function handleStop() {
    setCameraOn(false)
    speech.reset()
  }

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
            onReady={() => {}}
            onError={handleCameraError}
            onLipData={speech.feed}
          />
          <Controls
            cameraOn={cameraOn}
            calibrated={speech.calibrated}
            calibrating={speech.status === 'calibrating'}
            onStart={handleStart}
            onStop={handleStop}
            onCalibrate={speech.calibrate}
            onClear={handleClear}
          />
          {notice && <p className="text-xs text-accent">{notice}</p>}
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
        Week 4 · webcam + lip detection + speaking capture · backend not connected yet
      </footer>
    </div>
  )
}
