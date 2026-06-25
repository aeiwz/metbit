import type { Metadata } from 'next'
import ReportIssueButton from './components/ReportIssueButton'
import './globals.css'

export const metadata: Metadata = {
  title: 'metbit documentation',
  description: 'Documentation for the metbit NMR metabolomics Python package',
  icons: {
    icon: [
      { url: '/favicon.png', type: 'image/png', sizes: '32x32' },
      { url: '/logo/Metbit-logo-only.svg', type: 'image/svg+xml' },
      { url: '/logo/Metbit-logo-only.png', type: 'image/png' },
    ],
    apple: '/apple-touch-icon.png',
  },
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `(()=>{try{const m=(localStorage.getItem('theme')||'system');const d=window.matchMedia('(prefers-color-scheme: dark)').matches;document.documentElement.setAttribute('data-theme', m==='dark'?'dark':m==='light'?'light':(d?'dark':'light'));}catch(e){}})();`,
          }}
        />
      </head>
      <body>
        {children}
        <ReportIssueButton />
      </body>
    </html>
  )
}
