'use strict';

// Give each Excalidraw diagram (<figure class="excal">) its aspect ratio as
// --excal-ar, so source/css/figures.css can size the link and image to the
// picture itself (a portrait diagram capped at 85vh is exactly as wide as the
// picture, not letterboxed in the full column) before the lazy image loads.
const FIG = /<figure class="excal"([^>]*)>([\s\S]*?<img\b[^>]*?\bwidth="(\d+)"[^>]*?\bheight="(\d+)")/g;

hexo.extend.filter.register('after_post_render', data => {
  data.content = data.content.replace(FIG, (match, attrs, rest, w, h) => {
    if (/\bstyle=/.test(attrs) || !+w || !+h) return match;
    return `<figure class="excal"${attrs} style="--excal-ar: ${(w / h).toFixed(4)}">${rest}`;
  });
  return data;
});
