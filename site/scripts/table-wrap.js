'use strict';

// Wrap markdown tables in a scroll container (styled in source/css/site.css,
// "Post blocks"). The table keeps display: table, so its semantics survive,
// and fills the text column; a table wider than the column scrolls inside the
// wrapper, which is labelled here and made focusable by the theme's
// focusScrollRegions() (themes/aerlume/source/js/index.js) only while the
// table actually overflows, so keyboard users can scroll it.
// Only markdown tables (<table> followed by <thead>) are wrapped; code blocks
// render as <table><tr> inside figure.highlight and are left alone.
const TABLE = /<table>(\s*<thead>[\s\S]*?<\/table>)/g;

function label(html) {
  const head = /<thead>[\s\S]*?<\/thead>/.exec(html);
  if (!head) return 'Table';
  const cells = [...head[0].matchAll(/<th[^>]*>([\s\S]*?)<\/th>/g)]
    .map(m => m[1].replace(/<[^>]+>/g, '').replace(/\s+/g, ' ').trim())
    .filter(Boolean)
    .slice(0, 3);
  const text = cells.join(', ').replace(/"/g, '&quot;');
  return text ? `Table: ${text}` : 'Table';
}

hexo.extend.filter.register('after_post_render', data => {
  data.content = data.content.replace(TABLE, (match, rest) =>
    `<div class="table-wrap" role="region" aria-label="${label(rest)}"><table>${rest}</div>`);
  return data;
});
