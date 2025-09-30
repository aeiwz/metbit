import createMDX from '@next/mdx'

/** @type {import('next').NextConfig} */
const baseConfig = {
  reactStrictMode: true,
  pageExtensions: ['ts', 'tsx', 'js', 'jsx', 'md', 'mdx'],
}

const withMDX = createMDX({
  extension: /\.mdx?$/,
})

export default withMDX(baseConfig)

