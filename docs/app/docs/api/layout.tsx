import Link from 'next/link'
import { FiArrowLeft, FiBookOpen } from 'react-icons/fi'

export default function ApiLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="apiDoc prose" style={{ paddingTop: 12 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
        <Link href="/docs/api" aria-label="Back to API index" style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}>
          <FiArrowLeft aria-hidden /> Back to API
        </Link>
        <div style={{ display: 'inline-flex', alignItems: 'center', gap: 6, color: '#6b7280' }}>
          <FiBookOpen aria-hidden /> API Documentation
        </div>
      </div>
      {children}
    </div>
  )
}

