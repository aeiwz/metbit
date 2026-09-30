# metbit GUI identity

The “resolved m” is a continuous baseline rising through two rounded resonances. Their shared structure suggests reproducibility; the separate square suggests a sampled data value. The mark is deliberately stylized, rather than a chemical assignment.

- `metbit-gui-light.svg`: horizontal GUI logo for light interfaces.
- `metbit-gui-dark.svg`: horizontal GUI logo for dark interfaces.
- `metbit-wordmark-*.svg`: primary film/product lockup.
- `metbit-icon-*.svg`: transparent compact mark.
- `metbit-app-icon.svg`: 512 px application tile.
- `*-mono.svg`: single-color version using `currentColor`.

All SVG wordmarks contain actual vector paths, not font-dependent text. Inter is bundled under its OFL license in `web/public/fonts/LICENSE`. Run `python3 identity/generate_logo.py` to regenerate; this additionally requires `fonttools[woff]` (tested with local fontTools). PNG exports are rendered in Chromium by `render/export_identity.ts`.

Clear space: at least 30 units around the 172 × 120 mark. Minimum mark width: 24 px; prefer 32 px with the sampled-variable square. Do not stretch, rotate, add shadows or replace peak geometry with an ECG zigzag. Blue #3b82f6 is inherited from the local GUI; the film uses lighter #74b9ef for dark-background legibility.
