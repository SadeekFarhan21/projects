import type { CollectionEntry } from "astro:content";
import config from "@/config";

/** A post that has shipped, and so is guaranteed to carry a publication date. */
export type PublishedPost = CollectionEntry<"posts"> & {
  data: CollectionEntry<"posts">["data"] & { pubDatetime: Date };
};

/**
 * Determines whether a post is eligible to be listed/rendered.
 *
 * - Excludes drafts in production, and any post that has no `pubDatetime`
 * - In production, excludes scheduled posts until `pubDatetime` minus the configured margin
 * - In dev, always shows non-draft posts to make authoring easier
 * - In dev, also shows drafts for local review (set PUBLIC_HIDE_DRAFTS=1 to
 *   preview exactly what production will list)
 */
export function postFilter(
  post: CollectionEntry<"posts">
): post is PublishedPost {
  const { data } = post;
  const previewDrafts =
    import.meta.env.DEV && import.meta.env.PUBLIC_HIDE_DRAFTS !== "1";
  // Drafts carry no pubDatetime until publish, so preview them as dated now.
  if (previewDrafts && data.draft && !data.pubDatetime)
    data.pubDatetime = new Date();
  if (!data.pubDatetime || (data.draft && !previewDrafts)) return false;
  const isPublishTimePassed =
    Date.now() >
    new Date(data.pubDatetime).getTime() - config.posts.scheduledPostMargin;
  return import.meta.env.DEV || isPublishTimePassed;
}
