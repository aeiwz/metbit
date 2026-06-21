import createMDX from '@next/mdx'
import { dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const docsRoot = dirname(fileURLToPath(import.meta.url))

/** @type {import('next').NextConfig} */
const baseConfig = {
  reactStrictMode: true,
  pageExtensions: ['ts', 'tsx', 'js', 'jsx', 'md', 'mdx'],
  outputFileTracingIncludes: {
    '/docs/[version]/**': ['./content/generated/**/*.json'],
  },
  async redirects() {
    return [
      {
        source: '/docs/overview',
        destination: '/docs/v9.0.0',
        permanent: false,
      },
      {
        source: '/docs/getting-started',
        destination: '/docs/v9.0.0/getting-started',
        permanent: false,
      },
      {
        source: '/docs/api/:path*',
        destination: '/docs/v9.0.0/api',
        permanent: false,
      },
    ]
  },
  turbopack: {
    root: docsRoot,
  },
}

const withMDX = createMDX({
  extension: /\.mdx?$/,
})

export default withMDX(baseConfig)
