import type { Metadata } from 'next'
import './globals.css'
import ThemeToggle from './components/ThemeToggle'

export const metadata: Metadata = {
  title: 'Metbit Docs',
  description: 'Documentation for Metbit built with Next.js',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet" />
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
