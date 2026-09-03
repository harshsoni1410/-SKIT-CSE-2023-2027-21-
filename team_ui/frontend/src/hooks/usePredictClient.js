// Week 6 - React wrapper around PredictClient.
// Connects while `enabled` (camera on), exposes a predict() that returns a promise.
import { useEffect, useRef, useState } from 'react'
import { PredictClient } from '../lib/predictClient.js'

export function usePredictClient({ enabled } = {}) {
  const [connState, setConnState] = useState('closed') // connecting | open | closed | error
  const [modelKind, setModelKind] = useState(null) // 'stub' | 'model'

  const clientRef = useRef(null)
  if (clientRef.current === null) {
    clientRef.current = new PredictClient({
      onStatus: setConnState,
      onModelInfo: (info) => setModelKind(info.model),
    })
  }

  useEffect(() => {
    const client = clientRef.current
    if (enabled) client.connect()
    else client.disconnect()
  }, [enabled])

  useEffect(() => () => clientRef.current?.disconnect(), [])

  return {
    connState,
    modelKind,
    predict: (tensor) => clientRef.current.predict(tensor),
  }
}
