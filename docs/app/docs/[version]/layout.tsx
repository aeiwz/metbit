import { notFound } from 'next/navigation'

import VersionedDocsShell from '@/app/components/VersionedDocsShell'
import { getRelease, getSnapshot, releases } from '@/lib/versioned-docs'

export default async function VersionLayout({
  children,
  params,
}: {
  children: React.ReactNode
  params: Promise<{ version: string }>
}) {
  const { version } = await params
  const release = getRelease(decodeURIComponent(version))
  if (!release) notFound()
  const snapshot = getSnapshot(release)

  return (
    <VersionedDocsShell release={release} releases={releases} snapshot={snapshot}>
      {children}
    </VersionedDocsShell>
  )
}
