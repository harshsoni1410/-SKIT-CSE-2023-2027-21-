// Week 6 - WebSocket client for the LipSense backend (WS /ws/predict).
//
//   const client = new PredictClient({ onStatus, onModelInfo })
//   client.connect()
//   const { word, confidence, stub } = await client.predict(tensor)   // tensor from sequenceToTensor()
//
// One prediction is in flight at a time (an utterance completes, we ask, we wait).

import { WS_URL } from '../constants.js'

export class PredictClient {
  constructor({ url = WS_URL, onStatus, onModelInfo } = {}) {
    this.url = url
    this.onStatus = onStatus // 'connecting' | 'open' | 'closed' | 'error'
    this.onModelInfo = onModelInfo // { model: 'stub' | 'model' }
    this.ws = null
    this._pending = null
    this._open = false
  }

  get connected() {
    return this._open
  }

  connect() {
    if (this.ws && (this.ws.readyState === WebSocket.CONNECTING || this.ws.readyState === WebSocket.OPEN)) {
      return
    }
    this.onStatus?.('connecting')

    let ws
    try {
      ws = new WebSocket(this.url)
    } catch {
      this.onStatus?.('error')
      return
    }
    this.ws = ws

    ws.onopen = () => {
      this._open = true
      this.onStatus?.('open')
    }
    ws.onclose = () => {
      this._open = false
      this.onStatus?.('closed')
      this._reject('connection closed')
    }
    ws.onerror = () => {
      this.onStatus?.('error')
    }
    ws.onmessage = (ev) => {
      let msg
      try {
        msg = JSON.parse(ev.data)
      } catch {
        return
      }
      if (msg.type === 'ready') {
        this.onModelInfo?.({ model: msg.model })
      } else if (msg.type === 'prediction') {
        this._resolve(msg)
      } else if (msg.error) {
        this._reject(msg.error)
      }
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.onclose = null
      this.ws.onerror = null
      try {
        this.ws.close()
      } catch {
        /* ignore */
      }
      this.ws = null
    }
    this._open = false
    this._reject('disconnected')
  }

  /**
   * @param {{data: Float32Array, shape: number[]}} tensor
   * @returns {Promise<{word:string, confidence:number, stub:boolean}>}
   */
  predict(tensor, timeoutMs = 8000) {
    return new Promise((resolve, reject) => {
      if (!this._open) return reject(new Error('backend not connected'))
      if (this._pending) return reject(new Error('a prediction is already in flight'))

      const timer = setTimeout(() => {
        this._pending = null
        reject(new Error('prediction timed out'))
      }, timeoutMs)
      this._pending = { resolve, reject, timer }

      this.ws.send(
        JSON.stringify({ data: Array.from(tensor.data), shape: tensor.shape }),
      )
    })
  }

  _resolve(msg) {
    if (!this._pending) return
    clearTimeout(this._pending.timer)
    const { resolve } = this._pending
    this._pending = null
    resolve(msg)
  }

  _reject(reason) {
    if (!this._pending) return
    clearTimeout(this._pending.timer)
    const { reject } = this._pending
    this._pending = null
    reject(new Error(reason))
  }
}
