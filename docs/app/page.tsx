import { redirect } from 'next/navigation'

import { latestRelease } from '@/lib/versioned-docs'

export default function HomePage() {
  redirect(`/docs/${encodeURIComponent(latestRelease.tag)}`)
}
