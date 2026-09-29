#!/usr/bin/env node
// Publish drafts and ship them: `pnpm run publish` (or `npm run publish`).
//
//   pnpm run publish                  publish every draft, build, commit, push
//   pnpm run publish smolgrad ...     publish only the named drafts (file names without .md)
//   pnpm run publish --dry-run        list what would be published, change nothing
//   pnpm run publish --no-push        publish and commit, but do not push
//
// `pnpm publish` (without "run") is pnpm's npm-registry command, which refuses
// here because the package is private.
import { execFileSync } from "node:child_process";
import { existsSync, readdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const site = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const drafts = path.join(site, "source", "_drafts");
const args = process.argv.slice(2);
const dryRun = args.includes("--dry-run");
const noPush = args.includes("--no-push");
const wanted = args.filter(a => !a.startsWith("--")).map(a => a.replace(/\.md$/, ""));

const run = (cmd, argv, opts = {}) =>
  execFileSync(cmd, argv, { cwd: site, stdio: "inherit", ...opts });
const out = (cmd, argv, cwd = site) =>
  execFileSync(cmd, argv, { cwd, encoding: "utf8" }).trim();

const available = existsSync(drafts)
  ? readdirSync(drafts).filter(f => f.endsWith(".md")).map(f => f.slice(0, -3)).sort()
  : [];
const missing = wanted.filter(s => !available.includes(s));
if (missing.length) {
  console.error(`Not in source/_drafts: ${missing.join(", ")}`);
  console.error(`Drafts: ${available.join(", ") || "(none)"}`);
  process.exit(1);
}
const slugs = wanted.length ? wanted : available;
if (!slugs.length) {
  console.log("Nothing to publish: source/_drafts is empty.");
  process.exit(0);
}

console.log(`Publishing ${slugs.length} post${slugs.length > 1 ? "s" : ""}:`);
for (const s of slugs) console.log(`  - ${s}  ->  /posts/${s}/`);
if (dryRun) process.exit(0);

// Refuse to sweep unrelated work into the publish commit.
const dirty = out("git", ["status", "--porcelain", "--", "."])
  .split("\n")
  .filter(Boolean)
  .filter(l => !l.slice(3).startsWith("site/source/_drafts/") && !l.slice(3).startsWith("site/source/_posts/"));
if (dirty.length) {
  console.error("\nUncommitted changes outside the posts; commit or stash them first:");
  console.error(dirty.map(l => "  " + l).join("\n"));
  process.exit(1);
}

for (const s of slugs) run("npx", ["hexo", "publish", s]);

console.log("\nBuilding to check the site...");
run("npm", ["run", "build"]);

const repoRoot = out("git", ["rev-parse", "--show-toplevel"]);
const rel = p => path.relative(repoRoot, path.join(site, p));
// Stage the new posts plus the removal of each published draft that git tracks.
// Untracked drafts have nothing to stage (and naming a path git does not know
// makes `git add` fail); other drafts stay out of the commit.
const paths = [
  rel("source/_posts"),
  ...slugs.map(s => rel(`source/_drafts/${s}.md`)).filter(p => out("git", ["ls-files", "--", p], repoRoot)),
];
run("git", ["add", "-A", "--", ...paths], { cwd: repoRoot });
const msg =
  slugs.length === 1
    ? `Publish ${slugs[0]}`
    : `Publish ${slugs.length} posts\n\n${slugs.map(s => `- ${s}`).join("\n")}`;
// Commit only these paths, so changes staged elsewhere in the repo stay out. A
// draft that was staged but never committed is gone from the index now, and
// naming it would make `git commit` fail, so only drafts in HEAD are named.
const commitPaths = paths.filter(
  (p, i) => i === 0 || out("git", ["ls-tree", "--name-only", "HEAD", "--", p], repoRoot),
);
run("git", ["commit", "-m", msg, "--", ...commitPaths], { cwd: repoRoot });

if (noPush) {
  console.log("\nCommitted. Run `git push` to deploy.");
} else {
  run("git", ["push"], { cwd: repoRoot });
  console.log(`\nPushed. Vercel deploys in about a minute:`);
  for (const s of slugs) console.log(`  https://projects.farhansadeek.com/posts/${s}/`);
}
