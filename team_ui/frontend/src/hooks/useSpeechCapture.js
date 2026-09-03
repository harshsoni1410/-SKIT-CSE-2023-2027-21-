// Week 4 - React wrapper around SpeechCapture.
// App feeds it lip data every frame; it reports status and finished utterances.
import { useEffect, useRef, useState } from 'react'
import { SpeechCapture } from '../lib/speechCapture.js'

export function useSpeechCapture({ onUtterance } = {}) {
  const [status, setStatus] = useState('not_talking') // calibrating | not_talking | recording | processing
  const [calibrated, setCalibrated] = useState(false)

  const onUtteranceRef = useRef(onUtterance)
  useEffect(() => {
    onUtteranceRef.current = onUtterance
  }, [onUtterance])

  const captureRef = useRef(null)
  if (captureRef.current === null) {
    captureRef.current = new SpeechCapture({
      onStatus: (s) => {
        setStatus(s)
        setCalibrated(captureRef.current.calibrated)
      },
      onUtterance: (seq) => onUtteranceRef.current?.(seq),
    })
  }

  return {
    status,
    calibrated,
    feed: (data) => captureRef.current.feed(data),
    calibrate: () => captureRef.current.startCalibration(),
    ready: () => captureRef.current.ready(),
    reset: () => {
      captureRef.current.reset()
      setStatus('not_talking')
      setCalibrated(false)
    },
  }
}
