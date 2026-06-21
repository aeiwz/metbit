'use client'

import { useEffect, useSyncExternalStore } from 'react'
import { FiSun, FiMonitor, FiMoon } from 'react-icons/fi'

type Mode = 'light' | 'dark' | 'system'
type ThemeSnapshot = 'light' | 'dark' | 'system-light' | 'system-dark'

function getStoredMode(): Mode {
  const stored = localStorage.getItem('theme')
  return stored === 'light' || stored === 'dark' || stored === 'system' ? stored : 'system'
}

function getThemeSnapshot(): ThemeSnapshot {
  if (typeof window === 'undefined') return 'system-light'
  const stored = getStoredMode()
  if (stored === 'system') {
    const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches
    return isDark ? 'system-dark' : 'system-light'
  }
  return stored
}

function subscribe(onStoreChange: () => void) {
  if (typeof window === 'undefined') return () => {}

  const mq = window.matchMedia('(prefers-color-scheme: dark)')
  const onThemeChange = () => onStoreChange()

  window.addEventListener('storage', onThemeChange)
  window.addEventListener('themechange', onThemeChange)
  mq.addEventListener?.('change', onThemeChange)

  return () => {
    window.removeEventListener('storage', onThemeChange)
    window.removeEventListener('themechange', onThemeChange)
    mq.removeEventListener?.('change', onThemeChange)
  }
}

function snapshotToMode(snapshot: ThemeSnapshot): Mode {
  return snapshot === 'light' || snapshot === 'dark' ? snapshot : 'system'
}

function snapshotToAppliedTheme(snapshot: ThemeSnapshot): Exclude<Mode, 'system'> {
  if (snapshot === 'light' || snapshot === 'dark') return snapshot
  return snapshot === 'system-dark' ? 'dark' : 'light'
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
  const snapshot = useSyncExternalStore<ThemeSnapshot>(
    subscribe,
    getThemeSnapshot,
    () => 'system-light',
  )
  const mode = snapshotToMode(snapshot)

  useEffect(() => {
    applyTheme(snapshotToAppliedTheme(snapshot))
  }, [snapshot])

  const choose = (m: Mode) => {
    localStorage.setItem('theme', m)
    applyTheme(m)
    window.dispatchEvent(new Event('themechange'))
  }

  return (
    <div className="themeToggle" role="group" aria-label="Color theme">
      <button
        type="button"
        className={mode === 'light' ? 'active' : ''}
        onClick={() => choose('light')}
        aria-pressed={mode === 'light'}
        aria-label="Use light theme"
        title="Light theme"
      >
        <FiSun aria-hidden />
      </button>
      <button
        type="button"
        className={mode === 'system' ? 'active' : ''}
        onClick={() => choose('system')}
        aria-pressed={mode === 'system'}
        aria-label="Use system theme"
        title="System theme"
      >
        <FiMonitor aria-hidden />
      </button>
      <button
        type="button"
        className={mode === 'dark' ? 'active' : ''}
        onClick={() => choose('dark')}
        aria-pressed={mode === 'dark'}
        aria-label="Use dark theme"
        title="Dark theme"
      >
        <FiMoon aria-hidden />
      </button>
    </div>
  )
}
