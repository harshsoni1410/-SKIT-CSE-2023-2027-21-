import { useCallback, useState } from 'react'
import Header from './components/Header.jsx'
import WebcamView from './components/WebcamView.jsx'
import StatusBadge from './components/StatusBadge.jsx'
import PredictionCard from './components/PredictionCard.jsx'
import HistoryList from './components/HistoryList.jsx'
import Controls from './components/Controls.jsx'
import ErrorBanner from './components/ErrorBanner.jsx'
import ConnectionBadge from './components/ConnectionBadge.jsx'
import { useSpeechCapture } from './hooks/useSpeechCapture.js'
import { usePredictClient } from './hooks/usePredictClient.js'
import { sequenceToTensor } from './lib/speechCapture.js'
import { HISTORY_LIMIT } from './constants.js'

// State shape follows DESIGN.md "State (in App)".
//  Week 1 static · Week 2 webcam · Week 3 lip detection · Week 4 speaking capture
//  Week 6: an utterance -> sequenceToTensor -> backend WS -> { word, confidence } -> UI + history
export default function App() {
  const [cameraOn, setCameraOn] = useState(false)
  const [prediction, setPrediction] = useState(null)
  const [history, setHistory] = useState([])
  const [error, setError] = useState(null)

  const predictClient = usePredictClient({ enabled: cameraOn })

  const handleUtterance = useCallback(
    async (sequence) => {
      const tensor = sequenceToTensor(sequence)
      try {
        const res = await predictClient.predict(tensor)
        setPrediction({ word: res.word, confidence: res.confidence })
        setHistory((h) => [
          { word: res.word, confidence: res.confidence, time: new Date().toLocaleTimeString() },
          ...h,
        ].slice(0, HISTORY_LIMIT))
      } catch (err) {
        setError(
          err.message === 'backend not connected'
            ? 'Backend not reachable — start it: cd team_ui/backend && uvicorn main:app --port 8000'
            : `Prediction failed: ${err.message}`,
        )
      } finally {
        speech.ready()
      }
    },
    // predictClient.predict + speech.ready delegate to stable refs
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  )

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
          <ConnectionBadge connState={predictClient.connState} modelKind={predictClient.modelKind} />
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
        Week 6 · webcam → lip detection → speaking capture → backend prediction
      </footer>
    </div>
  )
}
