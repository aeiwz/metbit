import type { MDXComponents } from 'mdx/types'

export function useMDXComponents(components: MDXComponents): MDXComponents {
  return {
    h1: (props) => <h1 style={{ marginBottom: 12 }} {...props} />,
    h2: (props) => <h2 style={{ marginTop: 12, marginBottom: 8 }} {...props} />,
    ul: (props) => <ul style={{ paddingLeft: 18, listStyle: 'disc' }} {...props} />,
    li: (props) => <li style={{ margin: '6px 0' }} {...props} />,
    ...components,
  }
}

