import type { Metadata } from 'next'
import FeedbackForm from './FeedbackForm'
import ThemeToggle from '@/app/components/ThemeToggle'
import MetbitMark from '@/app/components/MetbitMark'
import Link from 'next/link'

export const metadata: Metadata = {
  title: 'Feedback - metbit',
  description: 'Report a bug, request a feature, or start a discussion about metbit.',
}

export default function FeedbackPage() {
  return (
    <div className="feedbackPage">
      <header className="feedbackHeader">
        <Link href="/" className="feedbackBrand" aria-label="metbit home">
          <MetbitMark />
        </Link>
        <ThemeToggle />
      </header>

      <main className="feedbackMain">
        <div className="feedbackHero">
          <h1>Feedback</h1>
          <p>Report a bug, request a feature, or start a discussion. Your report opens as a GitHub issue or discussion pre-filled with your details.</p>
        </div>
        <FeedbackForm />
      </main>
    </div>
  )
}
