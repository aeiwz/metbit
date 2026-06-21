import Image from 'next/image'

export default function MetbitMark() {
  return (
    <span className="metbitLogo" aria-hidden="true">
      <Image
        className="metbitLogoImage metbitLogoLight"
        src="/logo/Metbit-logo-dark-mode.svg"
        alt=""
        aria-hidden="true"
        fill
        sizes="126px"
        unoptimized
      />
      <Image
        className="metbitLogoImage metbitLogoDark"
        src="/logo/Metbit-logo-dark-mode.svg"
        alt=""
        aria-hidden="true"
        fill
        sizes="126px"
        unoptimized
      />
    </span>
  )
}
