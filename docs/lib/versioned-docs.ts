import 'server-only'

import fs from 'node:fs'
import path from 'node:path'

import manifestData from '@/content/generated/releases.json'

export type DocFunction = {
  name: string
  signature: string
  doc: string
}

export type DocClass = {
  name: string
  doc: string
  methods: DocFunction[]
}

export type DocModule = {
  name: string
  path: string
  category: string
  doc: string
  classes: DocClass[]
  functions: DocFunction[]
  parseError: string | null
}

export type ApiSnapshot = {
  rootExports: string[]
  modules: DocModule[]
}

export type Release = {
  tag: string
  version: string
  name: string
  publishedAt: string | null
  url: string
  body: string
  snapshot: string
}

type Manifest = {
  generatedAt: string | null
  latest: string
  releaseCount: number
  snapshotCount: number
  releases: Release[]
}

export const manifest = manifestData as Manifest
export const releases = manifest.releases
export const latestRelease = releases[0]

export function getRelease(tag: string): Release | undefined {
  return releases.find((release) => release.tag === tag)
}

export function getSnapshot(release: Release): ApiSnapshot {
  const snapshotPath = path.join(
    process.cwd(),
    'content',
    'generated',
    'snapshots',
    `${release.snapshot}.json`,
  )
  return JSON.parse(fs.readFileSync(snapshotPath, 'utf8')) as ApiSnapshot
}

export function getModule(snapshot: ApiSnapshot, slug: string[]): DocModule | undefined {
  const moduleName = slug.join('.')
  return snapshot.modules.find((module) => module.name === moduleName)
}

export function moduleHref(version: string, moduleName: string): string {
  return `/docs/${encodeURIComponent(version)}/api/${moduleName.split('.').map(encodeURIComponent).join('/')}`
}

export function releaseHref(tag: string): string {
  return `/docs/${encodeURIComponent(tag)}`
}

export function formatDate(value: string | null): string {
  if (!value) return 'Unknown date'
  return new Intl.DateTimeFormat('en', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    timeZone: 'UTC',
  }).format(new Date(value))
}

export function summary(doc: string, fallback: string): string {
  const firstParagraph = doc.split(/\n\s*\n/)[0]?.replace(/\s+/g, ' ').trim()
  return firstParagraph || fallback
}
