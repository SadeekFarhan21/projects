'use strict';

// Minutes to read a post: words of running text in the rendered body at 230
// words a minute, at least 1. Shared by the post header and the home page list
// so the two always agree.
// Only the prose a reader reads counts. Left out: the References section and
// everything after it, figures (diagrams, charts, code blocks), other code,
// math (each MathML token would count as a word), citation markers ([3]), and
// HTML entities (&nbsp; would count as "nbsp").
hexo.extend.helper.register('read_minutes', function (post) {
  const plain = String((post && post.content) || '')
    .replace(/<h2[^>]*id="references"[\s\S]*$/, ' ')
    .replace(/<(script|style|svg|figure|pre|math|eqn|eq)\b[\s\S]*?<\/\1>/g, ' ')
    .replace(/<a class="cite"[^>]*>[\s\S]*?<\/a>/g, ' ')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&(#x?[0-9a-f]+|[a-z]+);/gi, ' ');
  const words = (plain.match(/[A-Za-z0-9_'-]+/g) || []).length;
  return Math.max(1, Math.round(words / 230));
});
