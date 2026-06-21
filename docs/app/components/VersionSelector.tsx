'use client'

import { usePathname, useRouter } from 'next/navigation'

import type { Release } from '@/lib/versioned-docs'

export default function VersionSelector({
  releases,
  current,
}: {
  releases: Release[]
  current: string
}) {
  const pathname = usePathname()
  const router = useRouter()

  function selectVersion(tag: string) {
    const segments = pathname.split('/')
    if (segments[1] === 'docs' && segments[2]) {
      segments[2] = encodeURIComponent(tag)
      const stableSuffix = segments[3]
      if (stableSuffix === 'getting-started' || stableSuffix === 'releases' || stableSuffix === 'api' && segments.length === 4) {
        router.push(segments.join('/'))
      } else {
        router.push(`/docs/${encodeURIComponent(tag)}`)
      }
      return
    }
    router.push(`/docs/${encodeURIComponent(tag)}`)
  }

  return (
    <label className="versionControl">
      <span>Version</span>
      <select value={current} onChange={(event) => selectVersion(event.target.value)}>
        {releases.map((release) => (
          <option key={release.tag} value={release.tag}>
            {release.version}
          </option>
        ))}
      </select>
    </label>
  )
}
