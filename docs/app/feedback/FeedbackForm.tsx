'use client'

import { useEffect, useState } from 'react'
import { FiAlertCircle, FiArrowRight, FiHelpCircle, FiHash, FiMessageSquare, FiStar, FiUsers, FiZap } from 'react-icons/fi'

const REPO = 'aeiwz/metbit'
const BASE = `https://github.com/${REPO}`

type IssueType = 'bug' | 'feature' | 'discussion'

const TYPES = [
  { id: 'bug' as IssueType, icon: FiAlertCircle, label: 'Bug report', description: 'Something is broken or behaving unexpectedly' },
  { id: 'feature' as IssueType, icon: FiZap, label: 'Feature request', description: 'Suggest a new capability or improvement' },
  { id: 'discussion' as IssueType, icon: FiMessageSquare, label: 'Discussion', description: 'Ask a question or explore an idea' },
]

const DISCUSSION_CATEGORIES = [
  { icon: FiHash, label: 'General', slug: 'general', description: 'General chat about metbit' },
  { icon: FiHelpCircle, label: 'Q&A', slug: 'q-a', description: 'Ask how to use metbit or interpret results' },
  { icon: FiStar, label: 'Ideas', slug: 'ideas', description: 'Propose a direction or workflow improvement' },
  { icon: FiUsers, label: 'Show and tell', slug: 'show-and-tell', description: 'Share what you built or analysed with metbit' },
]

const BUG_AREAS = [
  'Data loading (Bruker reader, file import)',
  'Preprocessing (baseline, normalisation, scaling, alignment)',
  'Multivariate analysis (PCA, OPLS-DA)',
  'Statistics (fold change, p-value, VIP)',
  'Visualisation / Dash app',
  'Other',
]

function buildBugBody(f: BugFields): string {
  const areas = f.areas.length
    ? f.areas.map(a => a === 'Other' && f.otherDetail ? `- [x] Other: ${f.otherDetail}` : `- [x] ${a}`).join('\n')
    : '- [ ] (none selected)'
  return [
    `**Describe the bug**\n${f.description}`,
    `**Affected area**\n${areas}`,
    f.steps ? `**Steps to reproduce**\n${f.steps}` : '',
    f.expected ? `**Expected behaviour**\n${f.expected}` : '',
    f.code ? `**Minimal reproducible example**\n\`\`\`python\n${f.code}\n\`\`\`` : '',
    f.traceback ? `**Error / traceback**\n\`\`\`\n${f.traceback}\n\`\`\`` : '',
    [
      `**Environment**`,
      `- metbit version: ${f.metbitVersion || ''}`,
      `- Python version: ${f.pythonVersion || ''}`,
      `- OS: ${f.os || ''}${f.osVersion === 'Other' ? (f.osVersionOther ? ` ${f.osVersionOther}` : '') : f.osVersion ? ` ${f.osVersion}` : ''}`,
      `- Architecture: ${f.arch === 'other' ? (f.archOther || 'Other') : f.arch || ''}`,
    ].join('\n'),
  ].filter(Boolean).join('\n\n')
}

function buildFeatureBody(f: FeatureFields): string {
  return [
    `**Describe the feature**\n${f.description}`,
    f.motivation ? `**Why is this useful for NMR metabolomics workflows?**\n${f.motivation}` : '',
    f.api ? `**Suggested API**\n\`\`\`python\n${f.api}\n\`\`\`` : '',
    f.references ? `**References**\n${f.references}` : '',
  ].filter(Boolean).join('\n\n')
}

const OS_OPTIONS = [
  'macOS',
  'Windows',
  'Ubuntu',
  'Debian',
  'Fedora',
  'CentOS / RHEL',
  'Other Linux',
  'Other',
] as const

const OS_VERSIONS: Record<string, string[]> = {
  macOS: ['Tahoe (26)', 'Sequoia (15)', 'Sonoma (14)', 'Ventura (13)', 'Monterey (12)', 'Big Sur (11)'],
  Windows: ['Windows Server 2025', 'Windows 11', 'Windows Server 2022', 'Windows Server 2019', 'Windows 10'],
  Ubuntu: ['24.04 LTS (Noble)', '22.04 LTS (Jammy)', '20.04 LTS (Focal)', '18.04 LTS (Bionic)'],
  Debian: ['13 (Trixie)', '12 (Bookworm)', '11 (Bullseye)', '10 (Buster)'],
  Fedora: ['41', '40', '39', '38'],
  'CentOS / RHEL': ['RHEL 10 / CentOS Stream 10', 'RHEL 9 / CentOS Stream 9', 'RHEL 8 / CentOS Stream 8', 'CentOS 7'],
  'Other Linux': [],
  Other: [],
}

const ARCH_OPTIONS = [
  { value: 'x86_64', label: 'x86_64 (Intel / AMD)' },
  { value: 'arm64', label: 'ARM64 (Apple Silicon / AWS Graviton)' },
  { value: 'arm32', label: 'ARM32' },
  { value: 'other', label: 'Other' },
] as const

type ArchValue = typeof ARCH_OPTIONS[number]['value']

interface BugFields {
  title: string
  description: string
  areas: string[]
  otherDetail: string
  steps: string
  expected: string
  code: string
  traceback: string
  metbitVersion: string
  pythonVersion: string
  os: string
  osVersion: string
  osVersionOther: string
  arch: ArchValue | ''
  archOther: string
}

interface FeatureFields {
  title: string
  description: string
  motivation: string
  api: string
  references: string
}

const defaultBug: BugFields = { title: '', description: '', areas: [], otherDetail: '', steps: '', expected: '', code: '', traceback: '', metbitVersion: '', pythonVersion: '', os: '', osVersion: '', osVersionOther: '', arch: '', archOther: '' }
const defaultFeature: FeatureFields = { title: '', description: '', motivation: '', api: '', references: '' }

const MACOS_VERSION: Record<number, string> = { 26: 'Tahoe (26)', 15: 'Sequoia (15)', 14: 'Sonoma (14)', 13: 'Ventura (13)', 12: 'Monterey (12)', 11: 'Big Sur (11)' }
const macOSVersionLabel = (major: number) => MACOS_VERSION[major >= 26 ? 26 : major] ?? ''

async function detectDevice(): Promise<Partial<BugFields>> {
  const result: Partial<BugFields> = {}

  // Modern Client Hints API — Chromium only
  if ('userAgentData' in navigator) {
    try {
      const uad = (navigator as Navigator & { userAgentData: { getHighEntropyValues: (h: string[]) => Promise<Record<string, string>> } }).userAgentData
      const data = await uad.getHighEntropyValues(['platform', 'platformVersion', 'architecture'])
      const { platform, platformVersion, architecture } = data

      if (architecture === 'arm' || architecture === 'arm64') result.arch = 'arm64'
      else if (architecture === 'x86') result.arch = 'x86_64'

      if (platform === 'macOS') {
        result.os = 'macOS'
        const major = parseInt(platformVersion.split('.')[0])
        result.osVersion = macOSVersionLabel(major)
      } else if (platform === 'Windows') {
        result.os = 'Windows'
        // platformVersion major >= 13 = Windows 11
        result.osVersion = parseInt(platformVersion.split('.')[0]) >= 13 ? 'Windows 11' : 'Windows 10'
      } else if (platform === 'Linux') {
        result.os = 'Other Linux'
      }
      return result
    } catch { /* fall through to UA */ }
  }

  // UA string fallback — Safari, Firefox
  const ua = navigator.userAgent

  if (/aarch64|arm64/i.test(ua)) result.arch = 'arm64'
  else if (/armv7|arm(?!64)/i.test(ua)) result.arch = 'arm32'
  else if (/x86_64|Win64|WOW64|amd64/i.test(ua)) result.arch = 'x86_64'

  const macMatch = ua.match(/Mac OS X (\d+)[._](\d+)/)
  if (macMatch) {
    result.os = 'macOS'
    const major = parseInt(macMatch[1])
    // UA may still report 10_x for older versions; 10.16+ is Big Sur (11)
    const version = major === 10 ? (parseInt(macMatch[2]) >= 16 ? 11 : 0) : major
    result.osVersion = macOSVersionLabel(version)
  } else if (/Windows NT/i.test(ua)) {
    result.os = 'Windows'
    // Can't reliably distinguish Win 10 vs 11 from UA alone
  } else if (/Linux/i.test(ua)) {
    result.os = 'Other Linux'
  }

  return result
}

export default function FeedbackForm() {
  const [type, setType] = useState<IssueType>('bug')
  const [bug, setBug] = useState<BugFields>(defaultBug)
  const [feature, setFeature] = useState<FeatureFields>(defaultFeature)
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [autoDetected, setAutoDetected] = useState(false)

  useEffect(() => {
    detectDevice().then(detected => {
      if (Object.keys(detected).length) {
        setBug(b => ({ ...b, ...detected }))
        setAutoDetected(true)
      }
    })
  }, [])

  function toggleArea(area: string) {
    setBug(b => ({
      ...b,
      areas: b.areas.includes(area) ? b.areas.filter(a => a !== area) : [...b.areas, area],
    }))
  }

  function validate(): boolean {
    const e: Record<string, string> = {}
    if (type === 'bug') {
      if (!bug.title.trim()) e.title = 'Title is required.'
      if (!bug.description.trim()) e.description = 'Description is required.'
      if (bug.areas.includes('Other') && !bug.otherDetail.trim()) e.otherDetail = 'Please describe the affected area.'
    }
    if (type === 'feature') {
      if (!feature.title.trim()) e.title = 'Title is required.'
      if (!feature.description.trim()) e.description = 'Description is required.'
    }
    setErrors(e)
    if (Object.keys(e).length) {
      document.querySelector('.feedbackInputError')?.scrollIntoView({ behavior: 'smooth', block: 'center' })
      return false
    }
    return true
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!validate()) return
    let url: string
    if (type === 'bug') {
      url = `${BASE}/issues/new?template=bug_report.md&labels=bug&title=${encodeURIComponent(`[Bug] ${bug.title}`)}&body=${encodeURIComponent(buildBugBody(bug))}`
    } else {
      url = `${BASE}/issues/new?template=feature_request.md&labels=enhancement&title=${encodeURIComponent(`[Feature] ${feature.title}`)}&body=${encodeURIComponent(buildFeatureBody(feature))}`
    }
    window.open(url, '_blank', 'noopener,noreferrer')
  }

  const field = (key: string) => errors[key] ? ' feedbackInputError' : ''

  return (
    <div className="feedbackFormWrap">
      {/* Type selector */}
      <fieldset className="feedbackTypeGroup">
        <legend className="feedbackLabel">Type</legend>
        <div className="feedbackTypeCards">
          {TYPES.map(({ id, icon: Icon, label, description }) => (
            <label key={id} className={`feedbackTypeCard${type === id ? ' active' : ''}`}>
              <input type="radio" name="type" value={id} checked={type === id} onChange={() => { setType(id); setErrors({}) }} className="srOnly" />
              <span className="feedbackTypeIcon"><Icon aria-hidden /></span>
              <span className="feedbackTypeLabel">{label}</span>
              <span className="feedbackTypeDesc">{description}</span>
            </label>
          ))}
        </div>
      </fieldset>

      {/* Discussion: category cards, no form */}
      {type === 'discussion' && (
        <div className="discussionCategories">
          <p className="feedbackNote" style={{ marginBottom: 0 }}>Choose a topic to open GitHub Discussions. No form needed - start writing there directly.</p>
          {DISCUSSION_CATEGORIES.map(({ icon: Icon, label, slug, description }) => (
            <a
              key={slug}
              href={`${BASE}/discussions/new?category=${slug}`}
              target="_blank"
              rel="noopener noreferrer"
              className="discussionCategoryCard"
            >
              <span className="discussionCategoryIcon"><Icon aria-hidden /></span>
              <span className="discussionCategoryText">
                <span className="discussionCategoryLabel">{label}</span>
                <span className="discussionCategoryDesc">{description}</span>
              </span>
              <FiArrowRight aria-hidden className="discussionCategoryArrow" />
            </a>
          ))}
        </div>
      )}

      {/* Bug / Feature form */}
      {type !== 'discussion' && (
        <form className="feedbackForm" onSubmit={handleSubmit} noValidate>

          {/* Title */}
          <label className="feedbackField">
            <span className="feedbackLabel">Title <span className="feedbackRequired">*</span></span>
            <input
              className={`feedbackInput${field('title')}`}
              type="text"
              placeholder={type === 'bug' ? 'e.g. PCA scores plot fails with NaN values' : 'e.g. Add OPLS-DA permutation test'}
              value={type === 'bug' ? bug.title : feature.title}
              onChange={e => {
                const v = e.target.value
                if (type === 'bug') setBug(b => ({ ...b, title: v }))
                else setFeature(f => ({ ...f, title: v }))
                if (errors.title) setErrors(er => ({ ...er, title: '' }))
              }}
            />
            {errors.title && <span className="feedbackFieldError">{errors.title}</span>}
          </label>

          {/* Bug fields */}
          {type === 'bug' && (
            <>
              <label className="feedbackField">
                <span className="feedbackLabel">Describe the bug <span className="feedbackRequired">*</span></span>
                <textarea
                  className={`feedbackTextarea${field('description')}`}
                  rows={3}
                  placeholder="What went wrong?"
                  value={bug.description}
                  onChange={e => { setBug(b => ({ ...b, description: e.target.value })); if (errors.description) setErrors(er => ({ ...er, description: '' })) }}
                />
                {errors.description && <span className="feedbackFieldError">{errors.description}</span>}
              </label>

              <fieldset className="feedbackField">
                <legend className="feedbackLabel">Affected area</legend>
                <div className="feedbackCheckGrid">
                  {BUG_AREAS.map(area => (
                    <div key={area}>
                      <label className="feedbackCheck">
                        <input type="checkbox" checked={bug.areas.includes(area)} onChange={() => toggleArea(area)} />
                        <span>{area}</span>
                      </label>
                      {area === 'Other' && bug.areas.includes('Other') && (
                        <>
                          <input
                            className={`feedbackInput feedbackOtherInput${field('otherDetail')}`}
                            type="text"
                            placeholder="Describe the affected area"
                            value={bug.otherDetail}
                            onChange={e => { setBug(b => ({ ...b, otherDetail: e.target.value })); if (errors.otherDetail) setErrors(er => ({ ...er, otherDetail: '' })) }}
                            autoFocus
                          />
                          {errors.otherDetail && <span className="feedbackFieldError feedbackOtherError">{errors.otherDetail}</span>}
                        </>
                      )}
                    </div>
                  ))}
                </div>
              </fieldset>

              <label className="feedbackField">
                <span className="feedbackLabel">Steps to reproduce</span>
                <textarea className="feedbackTextarea" rows={3} placeholder="1. Load data with metbit.read_bruker(...)&#10;2. Call metbit.pca(...)&#10;3. See error" value={bug.steps} onChange={e => setBug(b => ({ ...b, steps: e.target.value }))} />
              </label>

              <label className="feedbackField">
                <span className="feedbackLabel">Expected behaviour</span>
                <textarea className="feedbackTextarea" rows={2} placeholder="What should have happened?" value={bug.expected} onChange={e => setBug(b => ({ ...b, expected: e.target.value }))} />
              </label>

              <label className="feedbackField">
                <span className="feedbackLabel">Minimal reproducible example</span>
                <textarea className="feedbackTextarea feedbackCode" rows={5} placeholder="import metbit&#10;&#10;# smallest code that triggers the bug" value={bug.code} onChange={e => setBug(b => ({ ...b, code: e.target.value }))} />
              </label>

              <label className="feedbackField">
                <span className="feedbackLabel">Error / traceback</span>
                <textarea className="feedbackTextarea feedbackCode" rows={4} placeholder="Paste the full traceback here" value={bug.traceback} onChange={e => setBug(b => ({ ...b, traceback: e.target.value }))} />
              </label>

              <div className="feedbackRow">
                <label className="feedbackField">
                  <span className="feedbackLabel">metbit version</span>
                  <input className="feedbackInput" type="text" placeholder="e.g. 0.4.1" value={bug.metbitVersion} onChange={e => setBug(b => ({ ...b, metbitVersion: e.target.value }))} />
                </label>
                <label className="feedbackField">
                  <span className="feedbackLabel">Python version</span>
                  <input className="feedbackInput" type="text" placeholder="e.g. 3.11.4" value={bug.pythonVersion} onChange={e => setBug(b => ({ ...b, pythonVersion: e.target.value }))} />
                </label>
              </div>

              {autoDetected && (
                <p className="feedbackAutoDetect">
                  Environment auto-detected from your browser. Edit if incorrect.
                </p>
              )}

              <div className="feedbackRow feedbackRow2">
                <label className="feedbackField">
                  <span className="feedbackLabel">OS</span>
                  <select
                    className="feedbackInput feedbackSelect"
                    value={bug.os}
                    onChange={e => setBug(b => ({ ...b, os: e.target.value, osVersion: '', osVersionOther: '' }))}
                  >
                    <option value="">Select OS…</option>
                    {OS_OPTIONS.map(o => <option key={o} value={o}>{o}</option>)}
                  </select>
                </label>
                <div className="feedbackField">
                  <span className="feedbackLabel">OS version</span>
                  {bug.os && OS_VERSIONS[bug.os]?.length ? (
                    <>
                      <select
                        className="feedbackInput feedbackSelect"
                        value={bug.osVersion}
                        onChange={e => setBug(b => ({ ...b, osVersion: e.target.value, osVersionOther: '' }))}
                      >
                        <option value="">Select version…</option>
                        {OS_VERSIONS[bug.os].map(v => <option key={v} value={v}>{v}</option>)}
                        <option value="Other">Other</option>
                      </select>
                      {bug.osVersion === 'Other' && (
                        <input
                          className="feedbackInput"
                          style={{ marginTop: 6 }}
                          type="text"
                          placeholder="Enter version"
                          value={bug.osVersionOther}
                          onChange={e => setBug(b => ({ ...b, osVersionOther: e.target.value }))}
                          autoFocus
                        />
                      )}
                    </>
                  ) : (
                    <input
                      className="feedbackInput"
                      type="text"
                      placeholder={bug.os ? 'Enter version' : 'Select OS first'}
                      disabled={!bug.os}
                      value={bug.osVersion}
                      onChange={e => setBug(b => ({ ...b, osVersion: e.target.value }))}
                    />
                  )}
                </div>
              </div>

              <fieldset className="feedbackField">
                <legend className="feedbackLabel">Architecture</legend>
                <div className="feedbackArchGrid">
                  {ARCH_OPTIONS.map(({ value, label }) => (
                    <label key={value} className={`feedbackArchChip${bug.arch === value ? ' active' : ''}`}>
                      <input type="radio" name="arch" value={value} checked={bug.arch === value} onChange={() => setBug(b => ({ ...b, arch: value, archOther: '' }))} className="srOnly" />
                      {label}
                    </label>
                  ))}
                </div>
                {bug.arch === 'other' && (
                  <input
                    className="feedbackInput"
                    style={{ marginTop: 8 }}
                    type="text"
                    placeholder="Describe your architecture"
                    value={bug.archOther}
                    onChange={e => setBug(b => ({ ...b, archOther: e.target.value }))}
                    autoFocus
                  />
                )}
              </fieldset>
            </>
          )}

          {/* Feature fields */}
          {type === 'feature' && (
            <>
              <label className="feedbackField">
                <span className="feedbackLabel">Describe the feature <span className="feedbackRequired">*</span></span>
                <textarea
                  className={`feedbackTextarea${field('description')}`}
                  rows={3}
                  placeholder="What should metbit be able to do?"
                  value={feature.description}
                  onChange={e => { setFeature(f => ({ ...f, description: e.target.value })); if (errors.description) setErrors(er => ({ ...er, description: '' })) }}
                />
                {errors.description && <span className="feedbackFieldError">{errors.description}</span>}
              </label>

              <label className="feedbackField">
                <span className="feedbackLabel">Why is this useful for NMR metabolomics?</span>
                <textarea className="feedbackTextarea" rows={3} placeholder="Scientific or practical motivation" value={feature.motivation} onChange={e => setFeature(f => ({ ...f, motivation: e.target.value }))} />
              </label>

              <label className="feedbackField">
                <span className="feedbackLabel">Suggested API</span>
                <textarea className="feedbackTextarea feedbackCode" rows={4} placeholder="result = metbit.NewFunction(data, param=value)" value={feature.api} onChange={e => setFeature(f => ({ ...f, api: e.target.value }))} />
              </label>

              <label className="feedbackField">
                <span className="feedbackLabel">References</span>
                <textarea className="feedbackTextarea" rows={2} placeholder="Papers, tools, or similar implementations" value={feature.references} onChange={e => setFeature(f => ({ ...f, references: e.target.value }))} />
              </label>
            </>
          )}

          <button type="submit" className="feedbackSubmit">
            Open on GitHub <FiArrowRight aria-hidden />
          </button>
          <p className="feedbackNote">
            Your details will pre-fill a GitHub issue. You can review and edit before submitting.
          </p>
        </form>
      )}
    </div>
  )
}
