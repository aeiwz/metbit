import { notFound } from 'next/navigation'

import { getRelease, getSnapshot } from '@/lib/versioned-docs'

export default async function GettingStarted({
  params,
}: {
  params: Promise<{ version: string }>
}) {
  const { version } = await params
  const release = getRelease(decodeURIComponent(version))
  if (!release) notFound()
  const snapshot = getSnapshot(release)
  const preferredExports = ['nmr_preprocessing', 'Normalization', 'pca', 'opls_da', 'STOCSY']
  const imports = preferredExports.filter((name) => snapshot.rootExports.includes(name))

  return (
    <article className="referencePage" id="top">
      <div className="breadcrumbs">Getting started / Installation</div>
      <div className="versionNotice">
        These instructions install the selected release: metbit {release.version}.
      </div>
      <h1>Installing metbit {release.version}</h1>
      <p className="pageLead">
        Use an isolated Python environment so scientific dependencies remain reproducible.
      </p>

      <section id="content">
        <h2>Install from PyPI</h2>
        <pre><code>{`python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "metbit==${release.version}"`}</code></pre>

        <h2>Verify the installation</h2>
        <pre><code>{`python -c "import metbit; print(metbit.__version__)"`}</code></pre>

        <h2>Import the public API</h2>
        {imports.length ? (
          <pre><code>{`from metbit import (\n${imports.map((name) => `    ${name},`).join('\n')}\n)`}</code></pre>
        ) : (
          <div className="versionWarning">
            This historical release does not expose a reliably detectable package-root API.
            Review the API index and source links before adapting modern examples.
          </div>
        )}
      </section>

      <section id="source">
        <h2>Historical compatibility</h2>
        <p>
          Old releases may require older Python or dependency versions. The generated API pages
          describe the selected tag, but they do not change the runtime constraints of that
          historical package.
        </p>
      </section>
    </article>
  )
}
