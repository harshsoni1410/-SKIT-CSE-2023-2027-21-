// Week 4 - speaking detection + fixed-length sequence buffer.
//
// Mirrors the logic in team_video_processing/data_collection/collect.py so the browser
// captures an utterance the same way the Python data-collection tool does:
//
//   calibrate "mouth closed" baseline  ->  ratio > open_threshold  => RECORDING
//   buffer lip crops while recording   ->  ratio < close_threshold for N frames => end
//   normalise the buffer to SEQ_LEN frames  ->  hand it off for prediction
//
// SEQ_LEN / crop size must match preprocessing/preprocess.py.

import { CROP_W, CROP_H } from './lipDetector.js'

export const SEQ_LEN = 22

const CALIB_FRAMES = 40 // frames sampled for the "mouth closed" baseline
const SILENCE_FRAMES = 7 // consecutive "closed" frames that end an utterance
const MIN_UTTER_FRAMES = 6 // shorter -> discarded (blink / noise)
const MAX_UTTER_FRAMES = 60 // safety cap

/**
 * Force a list of frames to exactly `seqLen`:
 *  - more  -> uniform sampling
 *  - fewer -> repeat the last frame
 * Same rule as preprocess.fix_sequence_length().
 */
export function fixSequenceLength(frames, seqLen = SEQ_LEN) {
  const n = frames.length
  if (n === 0) return []
  if (n === seqLen) return frames.slice()
  if (n > seqLen) {
    const out = []
    for (let i = 0; i < seqLen; i++) {
      out.push(frames[Math.round((i * (n - 1)) / (seqLen - 1))])
    }
    return out
  }
  return [...frames, ...Array(seqLen - n).fill(frames[n - 1])]
}

/**
 * Turn 22 lip-crop canvases into the model input tensor:
 *   Float32Array of length 22*80*112*3, values 0..1, RGB, row-major
 *   logical shape (1, 22, 80, 112, 3)  -- matches preprocess.frames_to_tensor().
 */
export function sequenceToTensor(canvases) {
  const frame = CROP_W * CROP_H * 3
  const data = new Float32Array(canvases.length * frame)
  const tmp = document.createElement('canvas')
  tmp.width = CROP_W
  tmp.height = CROP_H
  const ctx = tmp.getContext('2d', { willReadFrequently: true })

  canvases.forEach((canvas, f) => {
    ctx.drawImage(canvas, 0, 0, CROP_W, CROP_H)
    const { data: px } = ctx.getImageData(0, 0, CROP_W, CROP_H) // RGBA
    for (let i = 0, j = f * frame; i < CROP_W * CROP_H; i++) {
      data[j++] = px[i * 4] / 255 // R
      data[j++] = px[i * 4 + 1] / 255 // G
      data[j++] = px[i * 4 + 2] / 255 // B
    }
  })

  return { data, shape: [1, canvases.length, CROP_H, CROP_W, 3] }
}

function cloneCanvas(src) {
  const c = document.createElement('canvas')
  c.width = src.width
  c.height = src.height
  c.getContext('2d').drawImage(src, 0, 0)
  return c
}

const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length
const std = (a, m) => Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length)

export class SpeechCapture {
  constructor({ onStatus, onUtterance } = {}) {
    this.onStatus = onStatus
    this.onUtterance = onUtterance
    this.reset()
  }

  reset() {
    this._calibBuffer = []
    this._calibrating = false
    this.baseline = null
    this.openThreshold = null
    this.closeThreshold = null
    this._recording = false
    this._utterFrames = []
    this._silence = 0
    this._status = 'not_talking'
  }

  get calibrated() {
    return this.openThreshold != null
  }

  get status() {
    return this._status
  }

  startCalibration() {
    this._calibBuffer = []
    this._calibrating = true
    this._recording = false
    this._utterFrames = []
    this._setStatus('calibrating')
  }

  _setStatus(s) {
    if (s !== this._status) {
      this._status = s
      this.onStatus?.(s)
    }
  }

  _finishCalibration() {
    const arr = this._calibBuffer
    if (arr.length < CALIB_FRAMES / 2) return false
    const m = mean(arr)
    const spread = Math.max(0.04, 2.5 * std(arr, m))
    this.baseline = m
    this.openThreshold = m + spread
    this.closeThreshold = m + spread * 0.6
    this._calibrating = false
    this._setStatus('not_talking')
    return true
  }

  /** Feed one frame: { crop, ratio } from the lip detector, or null when no face. */
  feed(data) {
    if (this._calibrating) {
      if (data) {
        this._calibBuffer.push(data.ratio)
        if (this._calibBuffer.length >= CALIB_FRAMES) this._finishCalibration()
      }
      return
    }

    if (!this.calibrated) {
      this._setStatus('not_talking')
      return
    }

    if (!data) {
      if (this._recording) this._tickSilence()
      else this._setStatus('not_talking')
      return
    }

    const speaking = this._recording
      ? data.ratio > this.closeThreshold
      : data.ratio > this.openThreshold

    if (speaking) {
      if (!this._recording) {
        this._recording = true
        this._utterFrames = []
        this._silence = 0
      }
      this._utterFrames.push(cloneCanvas(data.crop))
      this._silence = 0
      this._setStatus('recording')
      if (this._utterFrames.length >= MAX_UTTER_FRAMES) this._endUtterance()
    } else if (this._recording) {
      this._tickSilence()
    } else {
      this._setStatus('not_talking')
    }
  }

  _tickSilence() {
    this._silence += 1
    if (this._silence >= SILENCE_FRAMES) this._endUtterance()
  }

  _endUtterance() {
    const frames = this._utterFrames
    this._recording = false
    this._utterFrames = []
    this._silence = 0

    if (frames.length >= MIN_UTTER_FRAMES) {
      this._setStatus('processing')
      this.onUtterance?.(fixSequenceLength(frames, SEQ_LEN))
    } else {
      this._setStatus('not_talking')
    }
  }

  /** Called after a prediction comes back, to accept the next utterance. */
  ready() {
    if (!this._recording) this._setStatus('not_talking')
  }
}
