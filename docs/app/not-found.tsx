import Link from 'next/link'
import { FiArrowLeft, FiHome, FiSearch } from 'react-icons/fi'
import { latestRelease } from '@/lib/versioned-docs'
import PingPongGame from './components/PingPongGame'

export const metadata = {
  title: '404 – Page not found | metbit docs',
}

export default function NotFound() {
  const tag = latestRelease?.tag ?? 'v9.1.0'

  return (
    <div className="notFoundPage">
      <div className="notFoundInner">
        {/* ── label ── */}
        <p className="notFoundLabel" aria-hidden>ERROR</p>

        {/* ── heading ── */}
        <h1 className="notFoundHeading">404</h1>
        <p className="notFoundTagline">This page bounced out of bounds.</p>

        {/* ── game ── */}
        <PingPongGame />

        {/* ── copy ── */}
        <p className="notFoundHelper">
          Try heading back to the docs or search for what you need.
        </p>

        {/* ── actions ── */}
        <nav className="notFoundActions" aria-label="Recovery navigation">
          <Link
            href={`/docs/${encodeURIComponent(tag)}/getting-started`}
            className="notFoundBtn primary"
          >
            <FiArrowLeft aria-hidden />
            Back to Docs
          </Link>
          <Link
            href={`/docs/${encodeURIComponent(tag)}`}
            className="notFoundBtn secondary"
          >
            <FiSearch aria-hidden />
            Search Docs
          </Link>
          <Link href="/" className="notFoundBtn secondary">
            <FiHome aria-hidden />
            Go Home
          </Link>
        </nav>
      </div>
    </div>
  )
}
