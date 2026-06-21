'use client'

import { useEffect, useState } from 'react'
import { FiDownload, FiGithub, FiPackage } from 'react-icons/fi'

type DownloadMetrics = {
  version: string
  pypiVersion: number | null
  githubVersion: number | null
  pypiAllTime: number | null
  pypiOverall180Days: number | null
  pypiLastMonth: number | null
  githubTotal: number | null
}

const formatter = new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 })

function formatCount(value: number | null): string {
  return value === null ? 'Unavailable' : formatter.format(value)
}

function Metric({
  icon,
  label,
  value,
  detail,
}: {
  icon: React.ReactNode
  label: string
  value: number | null
  detail: string
}) {
  return (
    <div className="downloadMetric">
      <span className="downloadMetricIcon">{icon}</span>
      <span>
        <strong>{formatCount(value)}</strong>
        <small>{label}</small>
        <em>{detail}</em>
      </span>
    </div>
  )
}

export default function DownloadFooter({
  releaseTag,
  version,
  isLatest,
}: {
  releaseTag: string
  version: string
  isLatest: boolean
}) {
  const [metrics, setMetrics] = useState<DownloadMetrics | null>(null)
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    const controller = new AbortController()
    fetch(`/api/downloads/${encodeURIComponent(releaseTag)}`, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error('Download metrics unavailable')
        return response.json() as Promise<DownloadMetrics>
      })
      .then(setMetrics)
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === 'AbortError') return
        setFailed(true)
      })
    return () => controller.abort()
  }, [releaseTag])

  return (
    <footer className="downloadFooter">
      <div className="downloadFooterIntro">
        <FiDownload aria-hidden />
        <div>
          <strong>{isLatest ? 'Project downloads' : `Downloads for metbit ${version}`}</strong>
          <span>
            PyPI and GitHub measure different distribution channels. Statistics refresh daily.
          </span>
        </div>
      </div>

      {!metrics && !failed ? (
        <div className="downloadMetricsLoading" aria-label="Loading download statistics">
          <span />
          <span />
          <span />
        </div>
      ) : failed ? (
        <p className="downloadMetricsUnavailable">Download statistics are temporarily unavailable.</p>
      ) : metrics ? (
        <div className={`downloadMetrics ${isLatest ? 'downloadMetricsLatest' : ''}`}>
          {isLatest && (
            <>
              <Metric
                icon={<FiPackage aria-hidden />}
                label="PyPI all-time"
                value={metrics.pypiAllTime}
                detail="Lifetime package total reported by Pepy"
              />
              <Metric
                icon={<FiPackage aria-hidden />}
                label="PyPI overall"
                value={metrics.pypiOverall180Days}
                detail="Retained 180-day total, mirrors excluded"
              />
              <Metric
                icon={<FiPackage aria-hidden />}
                label="PyPI downloads"
                value={metrics.pypiLastMonth}
                detail="Last 30 days, all versions"
              />
              <Metric
                icon={<FiGithub aria-hidden />}
                label="GitHub downloads"
                value={metrics.githubTotal}
                detail="All uploaded release assets"
              />
            </>
          )}
          <Metric
            icon={<FiPackage aria-hidden />}
            label={`PyPI ${version}`}
            value={metrics.pypiVersion}
            detail="Per-version total reported by ClickPy"
          />
          <Metric
            icon={<FiGithub aria-hidden />}
            label={`GitHub ${version}`}
            value={metrics.githubVersion}
            detail="Uploaded assets for this release"
          />
        </div>
      ) : null}

      <p className="downloadCaveat">
        Counts are distribution activity, not unique users. GitHub source archives and Git clones
        are not included. Sources:{' '}
        <a href="https://pypistats.org/packages/metbit">PyPI Stats</a>,{' '}
        <a href="https://pepy.tech/projects/metbit">Pepy</a>,{' '}
        <a href={`https://clickpy.clickhouse.com/dashboard/metbit?version=${encodeURIComponent(version)}`}>
          ClickPy
        </a>
        , and <a href={`https://github.com/aeiwz/metbit/releases/tag/${encodeURIComponent(releaseTag)}`}>GitHub Releases</a>.
      </p>
    </footer>
  )
}
