import Link from 'next/link'
import { notFound } from 'next/navigation'

import { formatDate, getRelease, releases } from '@/lib/versioned-docs'

export default async function ReleasesPage({
  params,
}: {
  params: Promise<{ version: string }>
}) {
  const { version } = await params
  const current = getRelease(decodeURIComponent(version))
  if (!current) notFound()

  return (
    <article className="referencePage" id="top">
      <div className="breadcrumbs">Documentation / Release history</div>
      <h1>Release history</h1>
      <p className="pageLead">
        All {releases.length} published GitHub releases are available as selectable documentation
        versions. Repeated API snapshots are stored once and reused.
      </p>
      <div className="releaseTable" id="content">
        {releases.map((release) => (
          <div className={release.tag === current.tag ? 'current' : ''} key={release.tag}>
            <Link href={`/docs/${encodeURIComponent(release.tag)}`}>
              <strong>{release.tag}</strong>
              <span>{formatDate(release.publishedAt)}</span>
            </Link>
            <a href={release.url}>GitHub release</a>
          </div>
        ))}
      </div>
      <section id="source">
        <h2>Documentation provenance</h2>
        <p>
          Release dates and links come from GitHub Releases. API signatures and docstrings come
          from each release’s exact Git tag.
        </p>
      </section>
    </article>
  )
}
