'use strict';

const fs = require('fs');
const path = require('path');

const supportedFont = file => /^[a-zA-Z0-9][a-zA-Z0-9._-]*\.(woff2?|ttf|otf)$/i.test(file);
const cssEscape = value => String(value).replace(/[\\']/g, '\\$&');

hexo.extend.helper.register('local_font', function () {
  const settings = this.theme.font || {};
  const requested = typeof settings.file === 'string' ? settings.file.trim() : 'auto';
  const family = typeof settings.family === 'string' && settings.family.trim()
    ? settings.family.trim()
    : 'Aerlume Local';
  const fontDirectory = path.resolve(this.view_dir, '..', 'source', 'fonts');

  let file = '';
  if (requested === 'auto' && fs.existsSync(fontDirectory)) {
    file = fs.readdirSync(fontDirectory).filter(supportedFont).sort()[0] || '';
  } else if (supportedFont(requested)) {
    file = requested;
  }

  if (!file || !fs.existsSync(path.join(fontDirectory, file))) return '';

  const extension = path.extname(file).slice(1).toLowerCase();
  const format = { woff: 'woff', woff2: 'woff2', ttf: 'truetype', otf: 'opentype' }[extension];
  const weight = /^\d{3}$/.test(String(settings.weight)) ? settings.weight : 400;
  const style = ['normal', 'italic', 'oblique'].includes(settings.style) ? settings.style : 'normal';
  const display = ['auto', 'block', 'fallback', 'optional', 'swap'].includes(settings.display) ? settings.display : 'swap';
  const escapedFamily = cssEscape(family);

  return `<style>@font-face{font-family:'${escapedFamily}';src:url('${this.url_for(`fonts/${file}`)}') format('${format}');font-style:${style};font-weight:${weight};font-display:${display}}:root{--aerlume-local-font:'${escapedFamily}'}</style>`;
});
