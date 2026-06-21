import Link from 'next/link'
import { notFound } from 'next/navigation'

import { getRelease, getSnapshot, moduleHref, summary } from '@/lib/versioned-docs'

export default async function ApiIndex({
  params,
}: {
  params: Promise<{ version: string }>
}) {
  const { version } = await params
  const release = getRelease(decodeURIComponent(version))
  if (!release) notFound()
  const snapshot = getSnapshot(release)
  const categories = Array.from(new Set(snapshot.modules.map((module) => module.category)))

  return (
    <article className="referencePage" id="top">
      <div className="breadcrumbs">Documentation / API reference</div>
      <div className="versionNotice">
        API snapshot for metbit {release.version}. Select another version in the top bar to compare.
      </div>
      <h1>API reference</h1>
      <p className="pageLead">
        Public classes and functions parsed directly from the Python source at {release.tag}.
      </p>

      <div id="content">
        {categories.map((category) => (
          <section key={category}>
            <h2>{category}</h2>
            <div className="apiIndexList">
              {snapshot.modules
                .filter((module) => module.category === category)
                .map((module) => (
                  <Link key={module.name} href={moduleHref(release.tag, module.name)}>
                    <code>{module.name}</code>
                    <span>{summary(module.doc, `${module.classes.length} classes · ${module.functions.length} functions`)}</span>
                  </Link>
                ))}
            </div>
          </section>
        ))}
      </div>

      <section id="source">
        <h2>Root exports</h2>
        {snapshot.rootExports.length ? (
          <div className="exportList">
            {snapshot.rootExports.map((name) => <code key={name}>{name}</code>)}
          </div>
        ) : (
          <p>No explicit package-root exports were detected for this release.</p>
        )}
      </section>
    </article>
  )
}
