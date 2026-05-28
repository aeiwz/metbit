# metbit documentation site

This is the Next.js documentation app for `metbit`. It uses the App Router and MDX pages for guides and API reference content.

## Structure

- `app/` — App Router
  - `page.tsx` — Landing page
  - `docs/` — Docs section with a sidebar layout
    - `overview/page.mdx`
    - `getting-started/page.mdx`
    - `api/` — API reference pages
- `globals.css` — global and docs layout styles
- `next.config.js` — MDX-enabled config
- `package.json` — scripts and dependencies

## Run locally

1. From repo root: `cd docs`
2. Install: `npm install`
3. Dev server: `npm run dev`
4. Open: http://localhost:3000

## Authoring notes

- Add new docs by creating folders under `app/docs/<slug>/page.mdx`.
- API pages live under `app/docs/api/<slug>/page.mdx`.
- Prefer root imports in examples, for example `from metbit import pca, opls_da`.
- Use subpackage imports only when documenting advanced internals, for example `from metbit.nmr.alignment import PeakAligner`.
- Keep code examples runnable with current package exports.
- MDX allows mixing Markdown with React components.
- Icons come from `react-icons`.

## Maintenance checklist

- Update quick-start examples when public imports change in `metbit/__init__.py`.
- Keep the API index aligned with public exports and important subpackage utilities.
- Run `npm run build` before publishing the docs site.
