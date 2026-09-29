# Hexo Theme Aerlume

[简体中文](README.zh-CN.md)

A lightweight Hexo theme focused on calm reading, clear typography, and a
distraction-free writing experience. This is a maintained derivative of the
original AirCloud theme and is released under the MIT License.

Original project:
[aircloud/hexo-theme-aircloud](https://github.com/aircloud/hexo-theme-aircloud)

## Quick Start

Install from npm:

```bash
cd your-hexo-site
npm install hexo-theme-aerlume --save
```

Or install directly from GitHub:

```bash
cd your-hexo-site/themes
git clone https://github.com/wu3227834/hexo-theme-aerlume.git hexo-theme-aerlume
cd ..
npm install hexo-generator-search --save
```

Enable the theme in your Hexo site's root `_config.yml`:

```yaml
theme: hexo-theme-aerlume

search:
  path: search.json
  field: post
```

## Demo Configuration

The repository includes a generic, privacy-safe Hexo root configuration:

[Open `demo/_config.yml`](https://github.com/wu3227834/hexo-theme-aerlume/blob/master/demo/_config.yml)

This file is a complete example of the **Hexo site's root `_config.yml`**. It
shows where users should configure the site title, author, domain, social
accounts, comments, analytics, search, pagination, feeds, and other site-level
settings. It is not a replacement for the theme's own `_config.yml`.

The template uses placeholders such as `example.com` and
`your-email@example.com`, so it does not expose any maintainer's personal
accounts or service identifiers. Copy it into a new Hexo site and replace the
placeholders:

```bash
cp themes/hexo-theme-aerlume/demo/_config.yml _config.yml
```

For an existing site, merge the relevant sections instead of overwriting the
whole configuration file. The example site name is `aerlume-journal`.

## Configuration Layers

- Root `_config.yml`: site-specific identity and services, including the
  author, email, domain, social links, Giscus, analytics, and donations.
- `themes/hexo-theme-aerlume/_config.yml`: reusable theme defaults, such as
  local font settings. It should not contain personal account information.

See [`README.zh-CN.md`](README.zh-CN.md) for the full configuration reference.

## Development

Edit Less sources under `source/_less/` and rebuild the compiled stylesheet:

```bash
npx lessc source/css/aerlume.less source/css/aerlume.css
```

Verify the theme in a Hexo site:

```bash
hexo clean
hexo generate
hexo server
```

## License and Attribution

This project is based on XiaotaoNie's AirCloud theme and is maintained under
the Aerlume name. When redistributing this project or substantial portions of
it, keep the original copyright notice and the full MIT License text in
[`LICENSE`](LICENSE).

Fonts, third-party scripts, icons, and other external assets may have separate
licenses and are not automatically covered by this theme's MIT License.
