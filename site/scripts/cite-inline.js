'use strict';

// Citation markers are written in posts as <sup>[[3]](#ref-3)</sup>. Render
// them inline at body size instead of superscript: each becomes
// <a class="cite" href="#ref-3">[3]</a> (styled in source/css/site.css,
// "Citations"). A run of adjacent markers ([2][3]) is joined to the preceding
// word by a no-break space, and within itself by word joiners, so it never
// wraps onto a line by itself or splits across lines.
// The list under the References heading gets class="references" so its
// numbers sit flush with the text column (source/css/site.css, after
// "Lists hang").
const RUN = /(?:<sup>\s*<a href="#ref-\d+">\[\d+\]<\/a>\s*<\/sup>)+/g;
const ONE = /<a href="#ref-(\d+)">\[\d+\]<\/a>/g;
const REFS = /(<h2 id="references">[\s\S]*?<\/h2>\s*)<ol>/;

hexo.extend.filter.register('after_post_render', data => {
  data.content = data.content
    .replace(RUN, (run, offset, html) => {
      const cites = [...run.matchAll(ONE)]
        .map(m => `<a class="cite" href="#ref-${m[1]}">[${m[1]}]</a>`)
        .join('&#8288;'); // word joiner: a run never breaks between markers
      const gap = /\s/.test(html[offset - 1] || ' ') ? '' : '&nbsp;';
      return gap + cites;
    })
    .replace(REFS, '$1<ol class="references">');
  return data;
});
