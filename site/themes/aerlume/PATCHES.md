# Local changes to hexo-theme-aerlume

Vendored from https://github.com/wu3227834/hexo-theme-aerlume at the commit in
UPSTREAM_COMMIT (the theme is not published on npm). MIT license kept in LICENSE.

- Removed the busuanzi visitor counter (script, post "Visit" count, footer stats).
- Removed the bundled TsangerYuYang font; `_config.aerlume.yml` turns the local font off.
- Translated hard-coded Chinese UI strings (labels, pager, tags page, 404, TOC, search, copy buttons) to English.
- Added WRITING and CODE nav links, driven by `writing_url` and `code_url` in the site `_config.yml`.
- On post pages, load KaTeX CSS and css/figures.css, and js/figures.js when a post contains `data-figure`.
- Dropped the 404 page's image, which the theme references but does not ship.
- Added css/site.css (loaded on every page) to undo the theme's word-break: break-all for English text.
- Fonts: Quicksand (text) and Fragment Mono (code) from Google Fonts, set in head.ejs and css/site.css.
