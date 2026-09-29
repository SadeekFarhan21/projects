'use strict';

// Build search.json from each post's rendered HTML as plain text, instead of
// hexo-generator-search's raw Markdown (which put **, `, $\frac..$, citation
// markers and <figure> tags into the search result snippets). Registered under
// the same generator name ('json') so it replaces the plugin's generator;
// scripts/ load after plugins.

const NAMED = { amp: '&', lt: '<', gt: '>', quot: '"', apos: "'", nbsp: ' ', hellip: '…', mdash: '—', ndash: '–', rsquo: '’', lsquo: '‘', rdquo: '”', ldquo: '“', times: '×', minus: '−' };

function unescapeHTML(s) {
  return s.replace(/&(#x[0-9a-f]+|#\d+|[a-z]+);/gi, (m, e) => {
    if (e[0] === '#') {
      const code = e[1] === 'x' || e[1] === 'X' ? parseInt(e.slice(2), 16) : parseInt(e.slice(1), 10);
      try { return String.fromCodePoint(code); } catch (err) { return m; }
    }
    return Object.prototype.hasOwnProperty.call(NAMED, e.toLowerCase()) ? NAMED[e.toLowerCase()] : m;
  });
}

// MathML elements that do not read sensibly once flattened to text
const COMPLEX_MATH = /<(mfrac|msqrt|mroot|msub|msup|msubsup|munder|mover|munderover|mtable|mmultiscripts)\b/;

function mathText(html) {
  return unescapeHTML(html.replace(/<[^>]+>/g, '')).replace(/\s+/g, ' ').trim();
}

function toPlainText(html) {
  let s = String(html || '');
  s = s
    // not text
    .replace(/<(script|style|svg|noscript|template)\b[\s\S]*?<\/\1>/gi, ' ')
    .replace(/<!--[\s\S]*?-->/g, ' ')
    // diagrams, charts and images: drop the whole block
    .replace(/<figure\b[\s\S]*?<\/figure>/gi, ' ')
    // citation markers: <a class="cite" href="#ref-1">[1]</a> (scripts/cite-inline.js)
    .replace(/(?:&nbsp;|&#8288;)?<a\b[^>]*href="#ref[^"]*"[^>]*>[\s\S]*?<\/a>/gi, '')
    // display math (texmath wraps it in <eqn>)
    .replace(/<eqn\b[^>]*>[\s\S]*?<\/eqn>/gi, ' [equation] ')
    // inline math: keep simple expressions as text, else a placeholder
    .replace(/<eq\b[^>]*>([\s\S]*?)<\/eq>/gi, (m, inner) => {
      const text = mathText(inner);
      return (!COMPLEX_MATH.test(inner) && text.length <= 24) ? text : '[math]';
    })
    .replace(/<math\b[\s\S]*?<\/math>/gi, ' [math] ')
    // block boundaries become spaces so words do not run together
    .replace(/<\/?(p|div|li|ul|ol|h[1-6]|tr|td|th|table|blockquote|pre|br|hr|section|dt|dd)\b[^>]*>/gi, ' ')
    .replace(/<[^>]+>/g, '');
  return unescapeHTML(s).replace(/\s+/g, ' ').trim();
}

function entry(config, item) {
  const out = {};
  if (item.title) out.title = item.title;
  if (item.path) out.url = config.root + item.path;
  if (config.search.content !== false) out.content = toPlainText(item.content);
  if (item.tags && item.tags.length) out.tags = item.tags.map(t => t.name);
  if (item.categories && item.categories.length) out.categories = item.categories.map(c => c.name);
  return out;
}

hexo.extend.generator.register('json', function (locals) {
  const config = this.config;
  const search = config.search || {};
  if (!search.path || !/\.json$/i.test(search.path)) return;
  const field = String(search.field || 'post').trim();
  const res = [];
  const keep = item => !(item.indexing !== undefined && !item.indexing);
  if (field !== 'page') locals.posts.sort('-date').filter(keep).forEach(p => res.push(entry(config, p)));
  if (field !== 'post') locals.pages.filter(keep).forEach(p => res.push(entry(config, p)));
  return { path: search.path, data: JSON.stringify(res) };
});
