'use strict';

// Tags shown on a post and in the post list: programming languages are left
// out (config.hidden_tags) and at most config.max_tags are shown. Every tag
// still exists and keeps its tag page; this only trims what is displayed.
hexo.extend.helper.register('display_tags', function (tags) {
  const hidden = new Set((this.config.hidden_tags || []).map(t => String(t).toLowerCase()));
  const max = this.config.max_tags || 3;
  const list = tags && typeof tags.toArray === 'function' ? tags.toArray() : (tags || []);
  return list.filter(tag => !hidden.has(String(tag.name).toLowerCase())).slice(0, max);
});
