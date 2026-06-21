'use client'

import { useMemo, useState } from 'react'
import { FiSearch, FiX } from 'react-icons/fi'

type SearchItem = {
  label: string
  detail: string
  href: string
}

export default function DocsSearch({ items }: { items: SearchItem[] }) {
  const [query, setQuery] = useState('')
  const results = useMemo(() => {
    const normalized = query.trim().toLowerCase()
    if (!normalized) return []
    return items
      .filter((item) => `${item.label} ${item.detail}`.toLowerCase().includes(normalized))
      .slice(0, 12)
  }, [items, query])

  return (
    <div className="docsSearch">
      <FiSearch aria-hidden />
      <input
        aria-label="Search documentation"
        placeholder="Search documentation"
        value={query}
        onChange={(event) => setQuery(event.target.value)}
      />
      {query ? (
        <button type="button" aria-label="Clear search" onClick={() => setQuery('')}>
          <FiX aria-hidden />
        </button>
      ) : (
        <kbd>/</kbd>
      )}
      {query && (
        <div className="searchResults">
          {results.length ? (
            results.map((result) => (
              <a key={result.href} href={result.href} onClick={() => setQuery('')}>
                <strong>{result.label}</strong>
                <span>{result.detail}</span>
              </a>
            ))
          ) : (
            <p>No documentation found.</p>
          )}
        </div>
      )}
    </div>
  )
}
