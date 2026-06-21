import { NextResponse } from 'next/server'

import { getDownloadMetrics } from '@/lib/download-metrics'

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ version: string }> },
) {
  const { version } = await params
  const metrics = await getDownloadMetrics(decodeURIComponent(version))

  if (!metrics) {
    return NextResponse.json({ error: 'Unknown metbit release' }, { status: 404 })
  }

  return NextResponse.json(metrics, {
    headers: {
      'Cache-Control': 'public, s-maxage=86400, stale-while-revalidate=604800',
    },
  })
}
