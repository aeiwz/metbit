"use client"
import { useEffect, useState } from 'react'
import { FiSun, FiMonitor, FiMoon } from 'react-icons/fi'

type Mode = 'light' | 'dark' | 'system'

function getStoredMode(): Mode {
  if (typeof window === 'undefined') return 'system'
  const stored = localStorage.getItem('theme')
  if (stored === 'light' || stored === 'dark' || stored === 'system') {
    return stored
  }
  return 'system'
}

function applyTheme(mode: Mode) {
  const root = document.documentElement
  if (mode === 'system') {
    root.setAttribute('data-theme', 'system')
    const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches
    root.setAttribute('data-theme', isDark ? 'dark' : 'light')
  } else {
    root.setAttribute('data-theme', mode)
  }
}

export default function ThemeToggle() {
  const [mode, setMode] = useState<Mode>(getStoredMode)

  useEffect(() => {
    const current = getStoredMode()
    applyTheme(current)

    const mq = window.matchMedia('(prefers-color-scheme: dark)')
    const onChange = () => {
      if (getStoredMode() === 'system') applyTheme('system')
    }
    mq.addEventListener?.('change', onChange)
    return () => mq.removeEventListener?.('change', onChange)
  }, [])

  const choose = (m: Mode) => {
    localStorage.setItem('theme', m)
    setMode(m)
    applyTheme(m)
  }

  return (
    <div className="themeToggle" aria-label="Theme">
      <div className="seg" role="group" aria-label="Theme toggle">
        <button
          className={mode==='light' ? 'active' : ''}
          onClick={() => choose('light')}
          aria-pressed={mode==='light'}
          aria-label="Light theme"
        >
          <FiSun aria-hidden /> Light
        </button>
        <button
          className={mode==='system' ? 'active' : ''}
          onClick={() => choose('system')}
          aria-pressed={mode==='system'}
          aria-label="System theme"
        >
          <FiMonitor aria-hidden /> System
        </button>
        <button
          className={mode==='dark' ? 'active' : ''}
          onClick={() => choose('dark')}
          aria-pressed={mode==='dark'}
          aria-label="Dark theme"
        >
          <FiMoon aria-hidden /> Dark
        </button>
      </div>
    </div>
  )
}
