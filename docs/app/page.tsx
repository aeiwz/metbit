import Link from 'next/link'
import { FiArrowRight, FiBookOpen, FiLayers, FiPackage, FiRefreshCw, FiTool } from 'react-icons/fi'

import MetbitMark from './components/MetbitMark'
import ThemeToggle from './components/ThemeToggle'
import { formatDate, latestRelease, manifest, releaseHref } from '@/lib/versioned-docs'

export const metadata = {
  title: 'metbit documentation',
  description: 'Versioned documentation and project overview for metbit.',
}

const highlights = [
  {
    icon: FiPackage,
    title: 'Package overview',
    body: 'NMR metabolomics preprocessing, multivariate analysis, statistics, and visualization in Python.',
  },
  {
    icon: FiLayers,
    title: 'Versioned docs',
    body: `Browse ${manifest.releaseCount} published releases backed by ${manifest.snapshotCount} API snapshots.`,
  },
  {
    icon: FiTool,
    title: 'Practical workflow',
    body: 'Read Bruker data, preprocess spectra, fit PCA or OPLS-DA, and inspect interactive outputs.',
  },
  {
    icon: FiRefreshCw,
    title: 'Release-aware',
    body: `Latest release: ${latestRelease.version} published ${formatDate(latestRelease.publishedAt)}.`,
  },
]

export default function HomePage() {
  return (
    <div className="homePage">
      <div className="homeThemeControl" aria-label="Landing page theme control">
        <div className="homeThemeControlText">
        </div>
        <ThemeToggle />
      </div>

      <section className="homeHero">
        <div className="homeHeroCopy">
          <div className="homeHeroMeta">
            <span className="homeKicker">metbit documentation</span>
            <span className="homeMetaValue">
              {manifest.releaseCount} releases · {manifest.snapshotCount} snapshots
            </span>
          </div>

          <h1>Project overview</h1>
          <p className="homeLead">
            metbit is a Python toolkit for NMR metabolomics workflows: ingest Bruker data,
            preprocess spectra, normalize and align peaks, build PCA and OPLS-DA models, and
            inspect the results through interactive visualizations and Dash apps.
          </p>

          <div className="homeActions">
            <Link className="btn" href={releaseHref(latestRelease.tag)}>
              Open latest docs <FiArrowRight aria-hidden />
            </Link>
            <Link
              className="btn secondary"
              href={`/docs/${encodeURIComponent(latestRelease.tag)}/getting-started`}
            >
              Get started
            </Link>
          </div>

          <div className="homeReleaseNote">
            <FiBookOpen aria-hidden />
            <span>
              Latest release {latestRelease.version}
              {latestRelease.publishedAt ? `, published ${formatDate(latestRelease.publishedAt)}` : ''}.
            </span>
          </div>
        </div>

        <div className="homeHeroVisual">
          <div className="homeHeroLogo" aria-hidden="true">
            <MetbitMark />
          </div>
          <div className="homeHeroPanel">
            <div className="homePanelRow">
              <span>Release count</span>
              <strong>{manifest.releaseCount}</strong>
            </div>
            <div className="homePanelRow">
              <span>Snapshot count</span>
              <strong>{manifest.snapshotCount}</strong>
            </div>
            <div className="homePanelRow">
              <span>Latest version</span>
              <strong>{latestRelease.version}</strong>
            </div>
          </div>
        </div>
      </section>

      <section className="homeHighlights" aria-label="Project highlights">
        {highlights.map((item) => {
          const Icon = item.icon
          return (
            <article className="homeCard" key={item.title}>
              <span className="homeCardIcon" aria-hidden="true">
                <Icon />
              </span>
              <h2>{item.title}</h2>
              <p>{item.body}</p>
            </article>
          )
        })}
      </section>

      <section className="homeOverview">
        <div className="homeOverviewCopy">
          <h2>What you can do here</h2>
          <ul className="homeList">
            <li>Read release-specific installation and API documentation.</li>
            <li>Compare historical versions from a single docs site.</li>
            <li>Follow the pipeline from preprocessing through modeling and exploration.</li>
            <li>Jump straight into the latest release or the release history archive.</li>
          </ul>
        </div>
        <div className="homeOverviewPanel">
          <div className="homePanelRow">
            <span>Latest version</span>
            <strong>{latestRelease.version}</strong>
          </div>
          <div className="homePanelRow">
            <span>Release count</span>
            <strong>{manifest.releaseCount}</strong>
          </div>
          <div className="homePanelRow">
            <span>Snapshot count</span>
            <strong>{manifest.snapshotCount}</strong>
          </div>
          <div className="homePanelRow">
            <span>Docs entry point</span>
            <strong>Versioned routes</strong>
          </div>
          <Link className="homePanelLink" href={releaseHref(latestRelease.tag)}>
            Browse the latest release <FiArrowRight aria-hidden />
          </Link>
        </div>
      </section>
    </div>
  )
}
