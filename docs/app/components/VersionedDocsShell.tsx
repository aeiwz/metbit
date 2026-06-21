import Link from 'next/link'
import { FiBookOpen, FiGithub, FiMenu, FiPackage } from 'react-icons/fi'

import type { ApiSnapshot, Release } from '@/lib/versioned-docs'
import { moduleHref } from '@/lib/versioned-docs'

import DocsSearch from './DocsSearch'
import DownloadFooter from './DownloadFooter'
import MetbitMark from './MetbitMark'
import ThemeToggle from './ThemeToggle'
import VersionSelector from './VersionSelector'

const categoryOrder = [
  'Analysis and models',
  'NMR and preprocessing',
  'Statistics and utilities',
  'Visualization and apps',
  'Other',
]

export default function VersionedDocsShell({
  release,
  releases,
  snapshot,
  children,
}: {
  release: Release
  releases: Release[]
  snapshot: ApiSnapshot
  children: React.ReactNode
}) {
  const modulesByCategory = categoryOrder
    .map((category) => ({
      category,
      modules: snapshot.modules.filter((module) => module.category === category),
    }))
    .filter((group) => group.modules.length)

  const searchItems = [
    {
      label: 'Installation',
      detail: `Install metbit ${release.version}`,
      href: `/docs/${encodeURIComponent(release.tag)}/getting-started`,
    },
    {
      label: 'Release history',
      detail: 'Browse all published versions',
      href: `/docs/${encodeURIComponent(release.tag)}/releases`,
    },
    ...snapshot.modules.map((module) => ({
      label: module.name,
      detail: module.category,
      href: moduleHref(release.tag, module.name),
    })),
  ]

  return (
    <div className="versionedDocs">
      <header className="docsTopbar">
        <Link
          className="docsBrand"
          href="/"
          aria-label="metbit documentation home"
        >
          <MetbitMark />
        </Link>
        <nav className="primaryDocsNav" aria-label="Primary documentation">
          <Link href={`/docs/${encodeURIComponent(release.tag)}/getting-started`}>Install</Link>
          <Link href={`/docs/${encodeURIComponent(release.tag)}`}>User Guide</Link>
          <Link href={`/docs/${encodeURIComponent(release.tag)}/api`}>API</Link>
          <Link href={`/docs/${encodeURIComponent(release.tag)}/releases`}>Release history</Link>
        </nav>
        <DocsSearch items={searchItems} />
        <VersionSelector releases={releases} current={release.tag} />
        <ThemeToggle />
        <a className="topbarIcon" href="https://github.com/aeiwz/metbit" aria-label="metbit on GitHub">
          <FiGithub aria-hidden />
        </a>
      </header>

      <div className="docsWorkspace">
        <aside className="docsSidebar">
          <div className="mobileSidebarLabel">
            <FiMenu aria-hidden /> Documentation
          </div>
          <nav aria-label="Documentation sections">
            <section>
              <h2>Getting started</h2>
              <Link href={`/docs/${encodeURIComponent(release.tag)}/getting-started`}>Installation</Link>
              <Link href={`/docs/${encodeURIComponent(release.tag)}`}>Overview</Link>
            </section>
            <section>
              <h2>API reference</h2>
              <Link href={`/docs/${encodeURIComponent(release.tag)}/api`}>API index</Link>
              {modulesByCategory.map((group) => (
                <details key={group.category} open={group.category !== 'Other'}>
                  <summary>{group.category}</summary>
                  {group.modules.map((module) => (
                    <Link key={module.name} href={moduleHref(release.tag, module.name)}>
                      {module.name.replace(/^metbit\./, '')}
                    </Link>
                  ))}
                </details>
              ))}
            </section>
            <section>
              <h2>Release history</h2>
              <Link href={`/docs/${encodeURIComponent(release.tag)}/releases`}>All releases</Link>
              <a href={release.url}>Release on GitHub</a>
            </section>
          </nav>
          <div className="sidebarMeta">
            <FiPackage aria-hidden />
            <span>{snapshot.modules.length} documented modules</span>
          </div>
        </aside>

        <main className="docsMain">
          {children}
          <DownloadFooter
            releaseTag={release.tag}
            version={release.version}
            isLatest={release.tag === releases[0]?.tag}
          />
        </main>

        <aside className="docsToc" aria-label="On this page">
          <h2>On this page</h2>
          <a href="#top">Overview</a>
          <a href="#content">Contents</a>
          <a href="#source">Source</a>
        </aside>
      </div>
      <div className="docsMobileFooter">
        <FiBookOpen aria-hidden /> metbit {release.version} documentation
      </div>
    </div>
  )
}
