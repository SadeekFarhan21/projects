/**
 * Browser entry for the D3 figures, bundled by esbuild into source/js/figures.js.
 *
 * figures.ts came from the Astro site, which re-rendered on "astro:page-load".
 * Hexo pages are plain documents, so fire that event once the DOM is ready.
 */
import "./figures";

const render = () => document.dispatchEvent(new Event("astro:page-load"));

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", render);
} else {
  render();
}
