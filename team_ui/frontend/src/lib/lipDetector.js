// Week 3 - in-browser lip detection with MediaPipe FaceLandmarker.
//
// Gives, per video frame:
//   - box   : lip bounding box in video pixels (for the overlay)
//   - crop  : an offscreen 112x80 canvas with the lip region (model input size)
//   - ratio : inner-lip opening / mouth width  (used for speaking detection in Week 4)
//
// The crop size + margin match team_video_processing/preprocessing/preprocess.py
// (FRAME_W=112, FRAME_H=80, CROP_MARGIN=0.35) so the browser pipeline and the
// Python pipeline feed the model the same kind of image.

import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision'

const VERSION = '0.10.35'
const WASM_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VERSION}/wasm`
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'

export const CROP_W = 112
export const CROP_H = 80
const CROP_MARGIN = 0.35

// MediaPipe FaceMesh lip contour vertices (outer + inner lip), from FACEMESH_LIPS.
export const LIP_INDICES = [
  61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
  185, 40, 39, 37, 0, 267, 269, 270, 409,
  78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
  191, 80, 81, 82, 13, 312, 311, 310, 415,
]

// inner-lip vertical pair + mouth corners, for the speaking ratio
const INNER_TOP = 13
const INNER_BOTTOM = 14
const CORNER_LEFT = 61
const CORNER_RIGHT = 291

export async function createLipDetector() {
  const fileset = await FilesetResolver.forVisionTasks(WASM_BASE)

  let landmarker
  try {
    landmarker = await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
      runningMode: 'VIDEO',
      numFaces: 1,
    })
  } catch {
    // some machines have no working WebGL - fall back to CPU
    landmarker = await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: 'CPU' },
      runningMode: 'VIDEO',
      numFaces: 1,
    })
  }

  const crop = document.createElement('canvas')
  crop.width = CROP_W
  crop.height = CROP_H
  const cropCtx = crop.getContext('2d', { willReadFrequently: true })

  return {
    /**
     * Run detection on one video frame.
     * @param {HTMLVideoElement} video
     * @param {number} timestampMs monotonically increasing (performance.now())
     * @returns {{box:{x1,y1,x2,y2}, crop:HTMLCanvasElement, ratio:number} | null}
     */
    detect(video, timestampMs) {
      const vw = video.videoWidth
      const vh = video.videoHeight
      if (!vw || !vh) return null

      const res = landmarker.detectForVideo(video, timestampMs)
      const faces = res?.faceLandmarks
      if (!faces || faces.length === 0) return null
      const lm = faces[0]

      let minX = 1
      let minY = 1
      let maxX = 0
      let maxY = 0
      for (const i of LIP_INDICES) {
        const p = lm[i]
        if (p.x < minX) minX = p.x
        if (p.y < minY) minY = p.y
        if (p.x > maxX) maxX = p.x
        if (p.y > maxY) maxY = p.y
      }

      let x1 = minX * vw
      let y1 = minY * vh
      let x2 = maxX * vw
      let y2 = maxY * vh
      const mx = (x2 - x1) * CROP_MARGIN
      const my = (y2 - y1) * CROP_MARGIN
      x1 = Math.max(0, x1 - mx)
      y1 = Math.max(0, y1 - my)
      x2 = Math.min(vw, x2 + mx)
      y2 = Math.min(vh, y2 + my)

      const bw = x2 - x1
      const bh = y2 - y1
      if (bw <= 2 || bh <= 2) return null

      cropCtx.drawImage(video, x1, y1, bw, bh, 0, 0, CROP_W, CROP_H)

      const top = lm[INNER_TOP]
      const bottom = lm[INNER_BOTTOM]
      const cl = lm[CORNER_LEFT]
      const cr = lm[CORNER_RIGHT]
      const vertical = Math.hypot((top.x - bottom.x) * vw, (top.y - bottom.y) * vh)
      const horizontal = Math.hypot((cl.x - cr.x) * vw, (cl.y - cr.y) * vh) + 1e-6

      return { box: { x1, y1, x2, y2 }, crop, ratio: vertical / horizontal }
    },

    close() {
      landmarker.close()
    },
  }
}
