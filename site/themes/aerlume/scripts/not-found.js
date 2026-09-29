'use strict';

// Render layout/404.ejs to /404.html, which static hosts (Vercel included)
// serve for unknown paths. Nothing in source/ produces it.
hexo.extend.generator.register('not_found', function () {
  return {
    path: '404.html',
    layout: ['404'],
    data: { title: 'Page not found', comments: false }
  };
});
