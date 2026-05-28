import createMDX from '@next/mdx'
import { dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const docsRoot = dirname(fileURLToPath(import.meta.url))

/** @type {import('next').NextConfig} */
const baseConfig = {
  reactStrictMode: true,
  pageExtensions: ['ts', 'tsx', 'js', 'jsx', 'md', 'mdx'],
  turbopack: {
    root: docsRoot,
  },
}

const withMDX = createMDX({
  extension: /\.mdx?$/,
})

export default withMDX(baseConfig)
