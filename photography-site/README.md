# photography-site

Source of the public gallery repo [SadeekFarhan21/PhotographyWebsite](https://github.com/SadeekFarhan21/PhotographyWebsite) (snapshot of HEAD `18dfdbb`), the site behind photos.farhansadeek.com per the repo's homepage field.

## Attribution and scope (read this first)

This is **not** an original engineering project. It is a personalized copy of [rampatra/photography](https://github.com/rampatra/photography), a Jekyll template that enhances the "Multiverse" design by [AJ / html5up](https://html5up.net). Template credit is in the upstream README ("Thanks to AJ for the website template which I enhanced for jekyll"). The package is GPL-3.0; the full license text is not copied here (see the original repo).

Farhan Sadeek's own changes (38 of 39 commits) are limited to:
- `_config.yml` personalization (title, bio, social links, EXIF tags shown)
- `CNAME`, favicon, photo curation (52 images) and resizing/format changes
- a Vercel Speed Insights snippet in `_includes/footer.html`, added through a Vercel[bot] PR (#2, the 39th commit)

Not copied: `vendor/` (4,835 tracked Jekyll gem files), all photos, minified/third-party JS and fonts (jQuery, skel, poptrox, exif.js, FontAwesome), lockfiles, upstream LICENSE and README, `ads.txt`.

## Layout

- `_config.yml`, `index.html`, `_layouts/`, `_includes/`: Jekyll site
- `gulpfile.js`: image resize (ImageMagick via gulp-image-resize), sass and uglify tasks (from the template)
- `assets/sass/`, `assets/js/main.js`: template styles and script
- `results/measurements.txt`: counts and image sizes, see below

## Measurements

Produced by running `git`, `du`, `identify` (ImageMagick) over the full clone, output saved in `results/measurements.txt`: 52 full images (32,588 KB) and 52 thumbs (6,964 KB), 39 commits from 2024-11-18 to 2026-09-04. The gulpfile targets 1024 px wide fulls and 512 px thumbs, but the committed images are of mixed sizes (mostly 1600 to 3024 px wide fulls, 1280 px wide thumbs), so the repo's images were not all produced by that task.

Could not run: `bundle exec jekyll build` failed on the machine's Ruby 2.6 (`Could not find gem 'http_parser.rb'`), so no build output is claimed. There are no tests.

## The more interesting related work

A separate private repo (a fork of Sam Becker's [exif-photo-blog](https://github.com/sambecker/exif-photo-blog) with location collections, an R2 migration and a photo-curation pipeline) is not included here.
