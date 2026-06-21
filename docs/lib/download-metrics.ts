import 'server-only'

import { unstable_cache } from 'next/cache'

import { latestRelease, releases } from './versioned-docs'

const CACHE_SECONDS = 60 * 60 * 24
const REQUEST_TIMEOUT_MS = 12_000

type PyPIRecentResponse = {
  data?: {
    last_month?: number
  }
}

type PyPIOverallResponse = {
  data?: Array<{
    category?: string
    downloads?: number
  }>
}

type GitHubAsset = {
  download_count?: number
}

type GitHubRelease = {
  assets?: GitHubAsset[]
}

export type DownloadMetrics = {
  version: string
  pypiVersion: number | null
  githubVersion: number | null
  pypiAllTime: number | null
  pypiOverall180Days: number | null
  pypiLastMonth: number | null
  githubTotal: number | null
  updatedAt: string
}

async function fetchText(url: string): Promise<string | null> {
  try {
    const request = fetch(url, {
      headers: { 'User-Agent': 'metbit-docs' },
      cache: 'no-store',
    }).then(async (response) => (response.ok ? response.text() : null))
    const timeout = new Promise<null>((resolve) => {
      setTimeout(() => resolve(null), REQUEST_TIMEOUT_MS)
    })
    return await Promise.race([request, timeout])
  } catch {
    return null
  }
}

async function fetchJson<T>(url: string): Promise<T | null> {
  const text = await fetchText(url)
  if (!text) return null
  try {
    return JSON.parse(text) as T
  } catch {
    return null
  }
}

function assetDownloads(release: GitHubRelease | null): number | null {
  if (!release?.assets) return null
  return release.assets.reduce((total, asset) => total + (asset.download_count ?? 0), 0)
}

function parseCompactNumber(value: string): number | null {
  const match = value.trim().match(/^([\d,.]+)\s*([KMB])?$/i)
  if (!match) return null

  const base = Number(match[1].replaceAll(',', ''))
  if (!Number.isFinite(base)) return null

  const suffix = match[2]?.toUpperCase()
  const multiplier =
    suffix === 'K' ? 1_000 : suffix === 'M' ? 1_000_000 : suffix === 'B' ? 1_000_000_000 : 1
  return Math.round(base * multiplier)
}

function parseClickPyTotal(html: string | null): number | null {
  if (!html) return null

  const plain = html.match(
    />([\d,.]+\s*[KMB]?)<\/p><p class="text-slate-200 md:text-center">total<\/p>/i,
  )
  const streamed = html.match(
    /\\"children\\":\\"([\d,.]+\s*[KMB]?)\\"\}\],\[\\"\$\\",\\"p\\",null,\{\\"className\\":\\"text-slate-200 md:text-center\\",\\"children\\":\\"total\\"/i,
  )
  const value = plain?.[1] ?? streamed?.[1]
  return value ? parseCompactNumber(value) : null
}

function parsePepyAllTime(html: string | null): number | null {
  if (!html) return null
  const match = html.match(/"name":"Total downloads","value":(\d+)/)
  if (!match) return null
  const parsed = Number(match[1])
  return Number.isFinite(parsed) ? parsed : null
}

async function fetchAllGitHubDownloads(): Promise<number | null> {
  const pages = await Promise.all(
    [1, 2, 3].map((page) =>
      fetchJson<GitHubRelease[]>(
        `https://api.github.com/repos/aeiwz/metbit/releases?per_page=100&page=${page}`,
      ),
    ),
  )
  if (pages.every((page) => page === null)) return null

  return pages
    .flatMap((page) => page ?? [])
    .reduce((total, release) => total + (assetDownloads(release) ?? 0), 0)
}

async function collectDownloadMetrics(tag: string): Promise<DownloadMetrics | null> {
  const release = releases.find((item) => item.tag === tag)
  if (!release) return null

  const isLatest = release.tag === latestRelease.tag
  const [
    pypiRecent,
    pypiOverall,
    clickPyVersionHtml,
    pepyProjectHtml,
    githubRelease,
    githubTotal,
  ] = await Promise.all([
    isLatest
      ? fetchJson<PyPIRecentResponse>('https://pypistats.org/api/packages/metbit/recent')
      : Promise.resolve(null),
    isLatest
      ? fetchJson<PyPIOverallResponse>(
          'https://pypistats.org/api/packages/metbit/overall?mirrors=false',
        )
      : Promise.resolve(null),
    fetchText(
      `https://clickpy.clickhouse.com/dashboard/metbit?version=${encodeURIComponent(release.version)}`,
    ),
    isLatest ? fetchText('https://pepy.tech/projects/metbit') : Promise.resolve(null),
    fetchJson<GitHubRelease>(
      `https://api.github.com/repos/aeiwz/metbit/releases/tags/${encodeURIComponent(release.tag)}`,
    ),
    isLatest ? fetchAllGitHubDownloads() : Promise.resolve(null),
  ])

  return {
    version: release.version,
    pypiVersion: parseClickPyTotal(clickPyVersionHtml),
    githubVersion: assetDownloads(githubRelease),
    pypiAllTime: parsePepyAllTime(pepyProjectHtml),
    pypiOverall180Days:
      pypiOverall?.data?.reduce(
        (total, item) =>
          total + (item.category === 'without_mirrors' ? (item.downloads ?? 0) : 0),
        0,
      ) ?? null,
    pypiLastMonth: pypiRecent?.data?.last_month ?? null,
    githubTotal,
    updatedAt: new Date().toISOString(),
  }
}

export const getDownloadMetrics = unstable_cache(
  collectDownloadMetrics,
  ['metbit-download-metrics-v2'],
  { revalidate: CACHE_SECONDS },
)
