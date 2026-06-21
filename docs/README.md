# metbit documentation site

This is the Next.js documentation app for `metbit`. It provides release-specific
guides and API reference pages generated from each published GitHub release.

## Structure

- `app/docs/[version]/` — version-aware documentation routes
- `content/generated/releases.json` — published release manifest
- `content/generated/snapshots/` — deduplicated API snapshots parsed from Git tags
- `scripts/sync_version_docs.py` — release and API documentation generator
- `globals.css` — global and docs layout styles
- `next.config.js` — MDX-enabled config
- `package.json` — scripts and dependencies

## Run locally

1. From repo root: `cd docs`
2. Install: `npm install`
3. Dev server: `npm run dev`
4. Open: http://localhost:3000

## Refresh versioned documentation

The sync script reads GitHub Releases and the corresponding local Git tags:

```bash
npm run docs:sync
```

Run `git fetch --tags` first when new releases have been published. The
generator stores identical parsed APIs once and maps every release to its
matching snapshot, keeping the Vercel deployment compact.

## Authoring notes

- New releases require a GitHub Release and a matching local Git tag.
- API pages are generated from source signatures and docstrings; improve the
  Python docstring when generated documentation is incomplete.
- Historical releases may not support current Python versions or dependencies.
- Icons come from `react-icons`.

## Maintenance checklist

- Run `npm run docs:sync` after publishing a release.
- Review the generated manifest and release count.
- Check one current and one historical version.
- Run `npm run build` before publishing the docs site.
