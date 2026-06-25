'use client'

import Link from 'next/link'
import { FiAlertCircle } from 'react-icons/fi'

export default function ReportIssueButton() {
  return (
    <Link
      href="/feedback"
      className="reportIssueBtn"
      aria-label="Report issue or give feedback"
      title="Feedback"
    >
      <FiAlertCircle aria-hidden />
      <span>Feedback</span>
    </Link>
  )
}
