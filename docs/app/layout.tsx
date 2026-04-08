import type { Metadata } from 'next'
import { Roboto } from 'next/font/google'
import './globals.css'
import ThemeToggle from './components/ThemeToggle'

const roboto = Roboto({
  subsets: ['latin'],
  weight: ['400', '700'],
  display: 'swap',
})

export const metadata: Metadata = {
  title: 'Metbit Docs',
  description: 'Documentation for Metbit built with Next.js',
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
      <body className={roboto.className}>
        {children}
        <ThemeToggle />
      </body>
    </html>
  )
}
