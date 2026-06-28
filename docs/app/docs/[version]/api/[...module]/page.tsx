import { notFound } from 'next/navigation'

import {
  getModule,
  getRelease,
  getSnapshot,
  summary,
} from '@/lib/versioned-docs'

// matches: "name: desc"  "name (type): desc"  "name : type"
const PARAM_RE = /^([A-Za-z_]\w*)\s*(?:\([^)]*\))?\s*:\s*(.+)/

function DocText({ value }: { value: string }) {
  if (!value) return null
  const lines = value.split('\n')
  const blocks: React.ReactNode[] = []
  let paragraph: string[] = []

  function flushParagraph() {
    const content = paragraph.join(' ').replace(/\s+/g, ' ').trim()
    if (content) blocks.push(<p key={`p-${blocks.length}`}>{content}</p>)
    paragraph = []
  }

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index].trim()
    const next = lines[index + 1]?.trim() || ''

    if (line && /^-{3,}$/.test(next)) {
      flushParagraph()
      blocks.push(<h4 key={`h-${blocks.length}`}>{line.replace(/:$/, '')}</h4>)
      index += 1
      continue
    }
    if (!line) { flushParagraph(); continue }

    // strip leading bullet (• or -) then try to match as a named parameter
    const isBullet = /^[•\-]/.test(line)
    const inner    = isBullet ? line.replace(/^[•\-]\s*/, '') : line
    const m        = PARAM_RE.exec(inner)

    if (m) {
      flushParagraph()
      blocks.push(
        <div className="docParameter" key={`d-${blocks.length}`}>
          <code>{m[1]}</code>
          <span>{m[2].trim()}</span>
        </div>,
      )
      continue
    }

    // plain bullet with no name:desc pattern → list item
    if (isBullet) {
      flushParagraph()
      blocks.push(<li key={`li-${blocks.length}`}>{inner}</li>)
      continue
    }

    paragraph.push(line)
  }
  flushParagraph()
  return <div className="docText">{blocks.slice(0, 80)}</div>
}

export default async function ModuleReference({
  params,
}: {
  params: Promise<{ version: string; module: string[] }>
}) {
  const { version, module: moduleSlug } = await params
  const release = getRelease(decodeURIComponent(version))
  if (!release) notFound()
  const snapshot = getSnapshot(release)
  const moduleDoc = getModule(snapshot, moduleSlug.map(decodeURIComponent))
  if (!moduleDoc) notFound()

  return (
    <article className="referencePage apiReference" id="top">
      <div className="breadcrumbs">API reference / {moduleDoc.category}</div>
      <div className="versionNotice">
        You are viewing the documentation for metbit {release.version}.
        <a href={release.url}> Change release context</a>
      </div>
      <h1>{moduleDoc.name}</h1>
      <p className="pageLead">
        {summary(moduleDoc.doc, `${moduleDoc.category} module in metbit ${release.version}.`)}
      </p>
      <div className="signatureBlock">
        <code>import {moduleDoc.name}</code>
      </div>

      <div id="content">
        {moduleDoc.parseError && (
          <div className="versionWarning">This historical source could not be fully parsed: {moduleDoc.parseError}</div>
        )}

        {moduleDoc.classes.length > 0 && (
          <section>
            <h2>Classes</h2>
            {moduleDoc.classes.map((item) => (
              <div className="apiObject" key={item.name}>
                <h3 id={item.name}>{item.name}</h3>
                <DocText value={item.doc} />
                {item.methods.length > 0 && (
                  <>
                    <h4>Methods</h4>
                    <div className="memberTable">
                      {item.methods.map((method) => (
                        <details key={method.signature}>
                          <summary><code>{method.signature}</code></summary>
                          <div><DocText value={method.doc} /></div>
                        </details>
                      ))}
                    </div>
                  </>
                )}
              </div>
            ))}
          </section>
        )}

        {moduleDoc.functions.length > 0 && (
          <section>
            <h2>Functions</h2>
            <div className="memberTable">
              {moduleDoc.functions.map((item) => (
                <details key={item.signature}>
                  <summary><code>{item.signature}</code></summary>
                  <div><DocText value={item.doc} /></div>
                </details>
              ))}
            </div>
          </section>
        )}
      </div>

      <section id="source">
        <h2>Source</h2>
        <a href={`https://github.com/aeiwz/metbit/blob/${encodeURIComponent(release.tag)}/${moduleDoc.path}`}>
          {moduleDoc.path} at {release.tag}
        </a>
      </section>
    </article>
  )
}
