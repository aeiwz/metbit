import Image from 'next/image'

type MetbitMarkProps = {
  variant?: 'full' | 'compact'
}

export default function MetbitMark({ variant = 'full' }: MetbitMarkProps) {
  const lightSrc = variant === 'compact' ? '/logo/Metbit-logo-only.svg' : '/logo/Metbit-logo-light-mode.svg'
  const darkSrc = variant === 'compact' ? '/logo/Metbit-logo-only.svg' : '/logo/Metbit-logo-dark-mode.svg'

  return (
    <span className="metbitLogo" aria-hidden="true">
      <Image
        className="metbitLogoImage metbitLogoLight"
        src={lightSrc}
        alt=""
        aria-hidden="true"
        fill
        sizes="126px"
        unoptimized
      />
      <Image
        className="metbitLogoImage metbitLogoDark"
        src={darkSrc}
        alt=""
        aria-hidden="true"
        fill
        sizes="126px"
        unoptimized
      />
    </span>
  )
}
