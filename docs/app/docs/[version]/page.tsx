import type { Metadata } from 'next'
import Link from 'next/link'
import { notFound } from 'next/navigation'
import { FiActivity, FiArrowRight, FiDatabase, FiLayers, FiSettings } from 'react-icons/fi'

import { formatDate, getRelease, getSnapshot, moduleHref } from '@/lib/versioned-docs'

export async function generateMetadata({
  params,
}: {
  params: Promise<{ version: string }>
}): Promise<Metadata> {
  const { version } = await params
  return { title: `metbit ${decodeURIComponent(version)} documentation` }
}

export default async function VersionOverview({
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
      <div className="breadcrumbs">Documentation / {release.tag}</div>
      <div className="versionNotice">
        You are viewing metbit {release.version}, published {formatDate(release.publishedAt)}.
        <a href={release.url}> View release</a>
      </div>
      <h1>metbit</h1>
      <p className="pageLead">
        NMR metabolomics preprocessing, multivariate analysis, statistics, and interactive
        visualization in Python.
      </p>
      <div className="quickActions">
        <Link className="docsPrimaryButton" href={`/docs/${encodeURIComponent(release.tag)}/getting-started`}>
          Get started <FiArrowRight aria-hidden />
        </Link>
        <Link className="docsSecondaryButton" href={`/docs/${encodeURIComponent(release.tag)}/api`}>
          Browse API
        </Link>
      </div>

      <section id="content">
        <h2>What metbit {release.version} includes</h2>
        <div className="docFeatureList">
          {[
            [FiDatabase, 'NMR data processing', 'Bruker FID ingestion, calibration, baseline correction, denoising, peak handling, and alignment.'],
            [FiActivity, 'Analysis and models', 'PCA, OPLS-DA, PLS, cross-validation, VIP scoring, STOCSY, and large-scale workflows.'],
            [FiLayers, 'Statistics', 'Normalization, univariate analysis, scaling, and reusable scientific utilities.'],
            [FiSettings, 'Interactive exploration', 'Plotly visualizations and Dash applications for annotation, STOCSY, and peak selection.'],
          ].map(([Icon, title, description]) => (
            <div className="docFeature" key={String(title)}>
              <Icon aria-hidden />
              <div>
                <h3>{String(title)}</h3>
                <p>{String(description)}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section>
        <h2>API snapshot</h2>
        <p>
          This release contains {snapshot.modules.length} documented modules across{' '}
          {categories.length} categories and {snapshot.rootExports.length} detected package-root
          exports.
        </p>
        <div className="moduleLinkList">
          {snapshot.modules.slice(0, 12).map((module) => (
            <Link key={module.name} href={moduleHref(release.tag, module.name)}>
              <code>{module.name}</code>
              <FiArrowRight aria-hidden />
            </Link>
          ))}
        </div>
      </section>

      <section id="source">
        <h2>Version source</h2>
        <p>
          API documentation is generated from the Python source stored at the exact Git tag{' '}
          <a href={`https://github.com/aeiwz/metbit/tree/${encodeURIComponent(release.tag)}`}>
            {release.tag}
          </a>
          .
        </p>
      </section>
    </article>
  )
}
