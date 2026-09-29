'use strict';

// Minutes to read a post: words in the rendered body (figures left out) at 230
// words a minute, at least 1. Shared by the post header and the home page list
// so the two always agree.
hexo.extend.helper.register('read_minutes', function (post) {
  const plain = String((post && post.content) || '')
    .replace(/<figure[\s\S]*?<\/figure>/g, ' ')
    .replace(/<[^>]+>/g, ' ');
  const words = (plain.match(/[A-Za-z0-9_'-]+/g) || []).length;
  return Math.max(1, Math.round(words / 230));
});
