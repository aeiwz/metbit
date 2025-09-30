Metbit Docs (Next.js)
=====================

This is a Next.js + React documentation app for Metbit, using the App Router and MDX for authoring content.

Structure
---------
- `app/` — App Router
  - `page.tsx` — Landing page
  - `docs/` — Docs section with a sidebar layout
    - `overview/page.mdx`
    - `getting-started/page.mdx`
- `globals.css` — global and docs layout styles
- `next.config.js` — MDX-enabled config
- `package.json` — scripts and dependencies

Run Locally
-----------
1. From repo root: `cd docs`
2. Install: `npm install`
3. Dev server: `npm run dev`
4. Open: http://localhost:3000

Notes
-----
- Add new docs by creating folders under `app/docs/<slug>/page.mdx`.
- MDX allows mixing markdown with React components.
- Icons come from `react-icons` for consistent visuals.

