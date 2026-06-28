'use client'

import { useCallback, useEffect, useRef } from 'react'

// ── game constants ────────────────────────────────────────────────────────────
const PW          = 10    // paddle width  (px, CSS coords)
const PH          = 52    // paddle height
const BS          = 8     // ball size
const GAP         = 24    // paddle-to-wall gap
const SPEED_INIT  = 3.2
const SPEED_MAX   = 6.2
const SPEED_INC   = 0.22
const AI_SPEED    = 2.6
const TRAIL_LEN   = 7

// ── palette (always dark – arcade aesthetic regardless of page theme) ─────────
const CLR = {
  bg:       '#070f1c',
  bgDim:    '#060d19',
  border:   'rgba(13, 141, 155, 0.18)',
  center:   'rgba(13, 141, 155, 0.20)',
  watermark:'rgba(13, 141, 155, 0.11)',
  paddle:   '#48c8cf',
  paddleGlo:'rgba(72, 200, 207, 0.45)',
  ball:     '#ffffff',
  ballGlo:  'rgba(255, 255, 255, 0.55)',
  trail0:   'rgba(255,255,255,0.28)',
  hint:     'rgba(72, 200, 207, 0.40)',
}

type State = {
  bx: number; by: number
  vx: number; vy: number
  lpy: number; rpy: number
  spd: number
  flash: number                         // frames remaining for hit flash
  trail: Array<{ x: number; y: number }>
}

// ── helpers ───────────────────────────────────────────────────────────────────
function mkState(w: number, h: number): State {
  const angle = (Math.random() * 50 - 25) * (Math.PI / 180)
  const dir   = Math.random() > 0.5 ? 1 : -1
  return {
    bx: w / 2, by: h / 2,
    vx: Math.cos(angle) * SPEED_INIT * dir,
    vy: Math.sin(angle) * SPEED_INIT,
    lpy: h / 2 - PH / 2,
    rpy: h / 2 - PH / 2,
    spd: SPEED_INIT,
    flash: 0,
    trail: [],
  }
}

function normalize(vx: number, vy: number, spd: number): [number, number] {
  const mag = Math.sqrt(vx * vx + vy * vy)
  return [(vx / mag) * spd, (vy / mag) * spd]
}

// ── component ─────────────────────────────────────────────────────────────────
export default function PingPongGame() {
  const canvasRef  = useRef<HTMLCanvasElement>(null)
  const stateRef   = useRef<State | null>(null)
  const rafRef     = useRef<number>(0)
  const wRef       = useRef(560)
  const hRef       = useRef(220)
  const reducedRef = useRef(false)
  const keys       = useRef({ up: false, down: false })

  // ── draw one frame ──────────────────────────────────────────────────────────
  const draw = useCallback((ctx: CanvasRenderingContext2D, s: State, w: number, h: number) => {
    // fill background with scanline effect
    ctx.fillStyle = CLR.bg
    ctx.fillRect(0, 0, w, h)
    ctx.fillStyle = CLR.bgDim
    for (let y = 1; y < h; y += 2) ctx.fillRect(0, y, w, 1)

    // faint "404" watermark
    ctx.save()
    ctx.fillStyle = CLR.watermark
    ctx.font      = `700 ${Math.min(w * 0.28, 108)}px ui-monospace, monospace`
    ctx.textAlign    = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillText('404', w / 2, h / 2)
    ctx.restore()

    // center dashed divider
    ctx.save()
    ctx.setLineDash([5, 5])
    ctx.strokeStyle = CLR.center
    ctx.lineWidth   = 1
    ctx.beginPath(); ctx.moveTo(w / 2, 0); ctx.lineTo(w / 2, h); ctx.stroke()
    ctx.setLineDash([])
    ctx.restore()

    // ── ball trail ──────────────────────────────────────────────────────────
    s.trail.forEach((pt, i) => {
      const a = (i / s.trail.length) * 0.28
      ctx.fillStyle = `rgba(255,255,255,${a.toFixed(3)})`
      const sz = BS * (i / s.trail.length) * 0.7
      ctx.fillRect(Math.round(pt.x - sz / 2), Math.round(pt.y - sz / 2), sz, sz)
    })

    // ── ball ────────────────────────────────────────────────────────────────
    const bsz = s.flash > 0 ? BS + 3 : BS
    if (s.flash > 0) {
      ctx.save()
      ctx.shadowColor = CLR.ballGlo
      ctx.shadowBlur  = 14
    }
    ctx.fillStyle = CLR.ball
    ctx.fillRect(Math.round(s.bx - bsz / 2), Math.round(s.by - bsz / 2), bsz, bsz)
    if (s.flash > 0) ctx.restore()

    // ── left paddle (player) ────────────────────────────────────────────────
    ctx.save()
    ctx.shadowColor = CLR.paddleGlo
    ctx.shadowBlur  = s.flash > 0 ? 18 : 10
    ctx.fillStyle   = CLR.paddle
    ctx.fillRect(GAP, Math.round(s.lpy), PW, PH)
    ctx.restore()
    // pixel "4" label beside paddle (outside the canvas left)
    ctx.fillStyle    = CLR.hint
    ctx.font         = '700 10px ui-monospace, monospace'
    ctx.textAlign    = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillText('4', GAP + PW + 9, Math.round(s.lpy) + PH / 2)

    // ── right paddle (AI) ───────────────────────────────────────────────────
    const rx = w - GAP - PW
    ctx.save()
    ctx.shadowColor = CLR.paddleGlo
    ctx.shadowBlur  = 10
    ctx.fillStyle   = CLR.paddle
    ctx.fillRect(rx, Math.round(s.rpy), PW, PH)
    ctx.restore()
    ctx.fillStyle    = CLR.hint
    ctx.font         = '700 10px ui-monospace, monospace'
    ctx.textAlign    = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillText('4', rx - 9, Math.round(s.rpy) + PH / 2)

    // ── control hint (first render) ─────────────────────────────────────────
    ctx.fillStyle    = CLR.hint
    ctx.font         = '400 9px ui-monospace, monospace'
    ctx.textAlign    = 'center'
    ctx.textBaseline = 'bottom'
    ctx.fillText('move mouse / touch · arrow keys when focused', w / 2, h - 5)
  }, [])

  // ── game tick ───────────────────────────────────────────────────────────────
  const tick = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const s = stateRef.current
    if (!s)  return
    const w = wRef.current
    const h = hRef.current

    // keyboard paddle
    if (keys.current.up)   s.lpy = Math.max(0,      s.lpy - 4)
    if (keys.current.down) s.lpy = Math.min(h - PH,  s.lpy + 4)

    // advance ball
    s.bx += s.vx
    s.by += s.vy

    // trail
    s.trail.push({ x: s.bx, y: s.by })
    if (s.trail.length > TRAIL_LEN) s.trail.shift()

    // top / bottom wall
    if (s.by - BS / 2 <= 0) {
      s.by = BS / 2; s.vy = Math.abs(s.vy)
    } else if (s.by + BS / 2 >= h) {
      s.by = h - BS / 2; s.vy = -Math.abs(s.vy)
    }

    // AI right paddle – tracks ball with speed cap & slight lag
    const rCenter = s.rpy + PH / 2
    const lag = 0.82                      // reduce to make AI easier to beat
    if (rCenter < s.by - 2) s.rpy = Math.min(h - PH, s.rpy + AI_SPEED * lag)
    else if (rCenter > s.by + 2) s.rpy = Math.max(0, s.rpy - AI_SPEED * lag)

    // ── left paddle collision ───────────────────────────────────────────────
    const lEdge = GAP + PW
    if (s.vx < 0 && s.bx - BS / 2 <= lEdge && s.bx > GAP &&
        s.by + BS / 2 >= s.lpy && s.by - BS / 2 <= s.lpy + PH) {
      s.bx = lEdge + BS / 2
      s.spd = Math.min(SPEED_MAX, s.spd + SPEED_INC)
      const rel = (s.by - (s.lpy + PH / 2)) / (PH / 2)  // -1 to 1
      s.vx = Math.abs(s.vx)
      s.vy = rel * s.spd * 0.75
      ;[s.vx, s.vy] = normalize(s.vx, s.vy, s.spd)
      s.flash = 5
    }

    // ── right paddle collision ──────────────────────────────────────────────
    const rEdge = w - GAP - PW
    if (s.vx > 0 && s.bx + BS / 2 >= rEdge && s.bx < w - GAP &&
        s.by + BS / 2 >= s.rpy && s.by - BS / 2 <= s.rpy + PH) {
      s.bx = rEdge - BS / 2
      s.spd = Math.min(SPEED_MAX, s.spd + SPEED_INC)
      const rel = (s.by - (s.rpy + PH / 2)) / (PH / 2)
      s.vx = -Math.abs(s.vx)
      s.vy = rel * s.spd * 0.75
      ;[s.vx, s.vy] = normalize(s.vx, s.vy, s.spd)
      s.flash = 5
    }

    if (s.flash > 0) s.flash--

    // ── out of bounds → reset ────────────────────────────────────────────────
    if (s.bx < -30 || s.bx > w + 30) {
      const next = mkState(w, h)
      // keep paddle positions so player doesn't lose their spot
      next.lpy = s.lpy
      next.rpy = s.rpy
      Object.assign(s, next)
    }

    draw(ctx, s, w, h)
    rafRef.current = requestAnimationFrame(tick)
  }, [draw])

  // ── setup ───────────────────────────────────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    reducedRef.current = window.matchMedia('(prefers-reduced-motion: reduce)').matches

    // DPR-aware resize
    const resize = () => {
      const dpr  = window.devicePixelRatio || 1
      const cssW = canvas.offsetWidth  || 560
      const cssH = canvas.offsetHeight || 220
      canvas.width  = Math.round(cssW * dpr)
      canvas.height = Math.round(cssH * dpr)
      const ctx = canvas.getContext('2d')!
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      wRef.current = cssW
      hRef.current = cssH
      if (!stateRef.current) {
        stateRef.current = mkState(cssW, cssH)
      } else {
        // clamp to new dimensions
        stateRef.current.lpy = Math.min(stateRef.current.lpy, cssH - PH)
        stateRef.current.rpy = Math.min(stateRef.current.rpy, cssH - PH)
      }
      if (reducedRef.current) {
        draw(ctx, stateRef.current, cssW, cssH)
      }
    }

    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(canvas)

    if (!reducedRef.current) {
      rafRef.current = requestAnimationFrame(tick)
    }

    // ── mouse ────────────────────────────────────────────────────────────────
    const onMouseMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect()
      const y = e.clientY - rect.top
      if (stateRef.current) {
        stateRef.current.lpy = Math.max(0, Math.min(hRef.current - PH, y - PH / 2))
      }
    }
    canvas.addEventListener('mousemove', onMouseMove)

    // ── touch ────────────────────────────────────────────────────────────────
    const onTouchMove = (e: TouchEvent) => {
      e.preventDefault()
      const rect = canvas.getBoundingClientRect()
      const y = e.touches[0].clientY - rect.top
      if (stateRef.current) {
        stateRef.current.lpy = Math.max(0, Math.min(hRef.current - PH, y - PH / 2))
      }
    }
    canvas.addEventListener('touchmove', onTouchMove, { passive: false })

    // ── keyboard (only when canvas is focused) ────────────────────────────────
    const onKeyDown = (e: KeyboardEvent) => {
      if (document.activeElement !== canvas) return
      if (e.key === 'ArrowUp')   { e.preventDefault(); keys.current.up   = true }
      if (e.key === 'ArrowDown') { e.preventDefault(); keys.current.down = true }
    }
    const onKeyUp = (e: KeyboardEvent) => {
      if (e.key === 'ArrowUp')   keys.current.up   = false
      if (e.key === 'ArrowDown') keys.current.down = false
    }
    window.addEventListener('keydown', onKeyDown)
    window.addEventListener('keyup',   onKeyUp)

    return () => {
      cancelAnimationFrame(rafRef.current)
      ro.disconnect()
      canvas.removeEventListener('mousemove', onMouseMove)
      canvas.removeEventListener('touchmove', onTouchMove)
      window.removeEventListener('keydown', onKeyDown)
      window.removeEventListener('keyup',   onKeyUp)
    }
  }, [tick, draw])

  return (
    <canvas
      ref={canvasRef}
      className="pingPongCanvas"
      role="img"
      aria-label="Decorative pixel-art ping-pong animation. The left paddle is player-controlled."
      tabIndex={0}
    />
  )
}
