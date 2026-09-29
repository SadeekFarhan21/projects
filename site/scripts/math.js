'use strict';

// Render $...$ and $$...$$ to MathML at build time with Temml (no client-side
// JS). markdown-it-texmath parses the delimiters; it wraps inline math in <eq>
// and display math in <section><eqn>, styled in source/css/math.css.
// Registered through the renderer hook rather than _config.yml's
// markdown.plugins list so the engine option can be passed.
// Note: Temml renders an unknown command as red text (color: #b22222) instead
// of throwing, so look for red in a new post's math after building.
const texmath = require('markdown-it-texmath');
const temml = require('temml');

hexo.extend.filter.register('markdown-it:renderer', md => {
  md.use(texmath, {
    engine: temml,
    delimiters: 'dollars',
    katexOptions: { throwOnError: false }
  });
});
