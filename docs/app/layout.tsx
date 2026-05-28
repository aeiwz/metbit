import type { Metadata } from 'next'
import './globals.css'
import ThemeToggle from './components/ThemeToggle'

export const metadata: Metadata = {
  title: 'metbit documentation',
  description: 'Documentation for the metbit NMR metabolomics Python package',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `(()=>{try{const m=(localStorage.getItem('theme')||'system');const d=window.matchMedia('(prefers-color-scheme: dark)').matches;document.documentElement.setAttribute('data-theme', m==='dark'?'dark':m==='light'?'light':(d?'dark':'light'));}catch(e){}})();`,
          }}
        />
      </head>
      <body>
        {children}
        <ThemeToggle />
      </body>
    </html>
  )
}
