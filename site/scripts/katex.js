'use strict';

// Render $...$ and $$...$$ with KaTeX at build time. The package's CommonJS
// export is { default: plugin }, which _config.yml's markdown.plugins list
// cannot unwrap, so it is registered through the renderer hook instead.
const katex = require('@vscode/markdown-it-katex');

hexo.extend.filter.register('markdown-it:renderer', md => {
  md.use(katex.default || katex, { throwOnError: false });
});
