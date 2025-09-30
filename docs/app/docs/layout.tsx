import Link from 'next/link'
import { FiHome, FiBookOpen, FiFileText } from 'react-icons/fi'

export default function DocsLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="docsLayout">
      <aside className="sidebar">
        <nav aria-label="Docs Navigation">
          <ul>
            <li>
              <Link href="/" aria-label="Home">
                <FiHome className="icon" aria-hidden /> Home
              </Link>
            </li>
            <li>
              <Link href="/docs/overview">
                <FiBookOpen className="icon" aria-hidden /> Overview
              </Link>
            </li>
            <li>
              <Link href="/docs/getting-started">
                <FiFileText className="icon" aria-hidden /> Getting Started
              </Link>
            </li>
            <li>
              <Link href="/docs/api">
                <FiFileText className="icon" aria-hidden /> API Reference
              </Link>
            </li>
          </ul>
        </nav>
      </aside>
      <section className="content">
        <article>{children}</article>
      </section>
    </div>
  )
}

