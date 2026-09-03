import { useEffect, useRef, useState } from 'react'
import { createLipDetector, CROP_W, CROP_H } from '../lib/lipDetector.js'

// Week 2: live webcam via getUserMedia + canvas draw loop.
// Week 3: MediaPipe FaceLandmarker runs on each frame -> lip ROI box drawn on the
//         canvas, a 112x80 lip crop produced, and { crop, ratio } handed up via
//         onLipData for Week 4 (speaking detection + sequence buffer).
export default function WebcamView({ cameraOn, onError, onReady, onLipData }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const streamRef = useRef(null)
  const rafRef = useRef(0)
  const detectorRef = useRef(null)
  const onLipDataRef = useRef(onLipData)
  const [starting, setStarting] = useState(false)
  const [detectorState, setDetectorState] = useState('off') // off | loading | ready | error

  useEffect(() => {
    onLipDataRef.current = onLipData
  }, [onLipData])

  useEffect(() => {
    if (!cameraOn) {
      teardown()
      return
    }

    let cancelled = false
    setStarting(true)

    navigator.mediaDevices
      .getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' }, audio: false })
      .then(async (stream) => {
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop())
          return
        }
        streamRef.current = stream
        const video = videoRef.current
        if (video) {
          video.srcObject = stream
          await video.play().catch(() => {})
        }
        setStarting(false)
        onReady?.()
        startDrawLoop()
        initDetector(cancelled)
      })
      .catch((err) => {
        if (cancelled) return
        setStarting(false)
        onError?.(describeMediaError(err))
      })

    return () => {
      cancelled = true
      teardown()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cameraOn])

  async function initDetector() {
    if (detectorRef.current) return
    setDetectorState('loading')
    try {
      detectorRef.current = await createLipDetector()
      setDetectorState('ready')
    } catch (err) {
      console.error('lip detector failed to load', err)
      setDetectorState('error')
      onError?.('Lip detection model could not load (check your internet connection).')
    }
  }

  function startDrawLoop() {
    const tick = () => {
      const video = videoRef.current
      const canvas = canvasRef.current
      if (video && canvas && video.readyState >= 2) {
        if (canvas.width !== video.videoWidth) {
          canvas.width = video.videoWidth
          canvas.height = video.videoHeight
        }
        const ctx = canvas.getContext('2d')
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

        const detector = detectorRef.current
        if (detector) {
          const result = detector.detect(video, performance.now())
          if (result) {
            drawLipBox(ctx, result.box)
            drawCropPreview(ctx, result.crop, canvas.width)
            onLipDataRef.current?.({ crop: result.crop, ratio: result.ratio, t: Date.now() })
          } else {
            onLipDataRef.current?.(null)
          }
        }
      }
      rafRef.current = requestAnimationFrame(tick)
    }
    cancelAnimationFrame(rafRef.current)
    rafRef.current = requestAnimationFrame(tick)
  }

  function teardown() {
    cancelAnimationFrame(rafRef.current)
    if (detectorRef.current) {
      detectorRef.current.close()
      detectorRef.current = null
    }
    setDetectorState('off')
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop())
      streamRef.current = null
    }
    if (videoRef.current) videoRef.current.srcObject = null
  }

  return (
    <div className="relative aspect-video w-full overflow-hidden rounded-xl border border-line bg-black">
      <video ref={videoRef} className="hidden" playsInline muted />
      <canvas ref={canvasRef} className="h-full w-full object-cover" />

      {!cameraOn && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-slate-500">
          <svg className="h-10 w-10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <rect x="2" y="6" width="14" height="12" rx="2" />
            <path d="M16 10l6-3v10l-6-3" />
          </svg>
          <span className="text-sm">Camera is off</span>
        </div>
      )}

      {cameraOn && starting && (
        <div className="absolute inset-0 flex items-center justify-center text-sm text-slate-400">
          starting camera…
        </div>
      )}

      {cameraOn && !starting && detectorState === 'loading' && (
        <div className="absolute left-2 top-2 rounded bg-black/60 px-2 py-1 text-xs text-slate-300">
          loading lip model…
        </div>
      )}
    </div>
  )
}

function drawLipBox(ctx, box) {
  ctx.save()
  ctx.strokeStyle = '#2dd4bf'
  ctx.lineWidth = 2
  ctx.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1)
  ctx.restore()
}

function drawCropPreview(ctx, cropCanvas, canvasWidth) {
  const scale = 2
  const w = CROP_W * scale
  const h = CROP_H * scale
  const x = canvasWidth - w - 10
  const y = 10
  ctx.save()
  ctx.strokeStyle = '#2dd4bf'
  ctx.lineWidth = 1
  ctx.drawImage(cropCanvas, x, y, w, h)
  ctx.strokeRect(x, y, w, h)
  ctx.restore()
}

function describeMediaError(err) {
  switch (err?.name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return 'Camera permission denied. Allow camera access in the browser and try again.'
    case 'NotFoundError':
    case 'DevicesNotFoundError':
      return 'No camera found. Connect a webcam and try again.'
    case 'NotReadableError':
    case 'TrackStartError':
      return 'Camera is already in use by another app. Close it and try again.'
    default:
      return `Could not start the camera (${err?.name || 'unknown error'}).`
  }
}
