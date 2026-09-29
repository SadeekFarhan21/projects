"use client";
import { get, set, del, keys } from "idb-keyval";
import type { VaultEntry } from "./types";

const PREFIX = "vault:";

export async function saveEntry(entry: Omit<VaultEntry, "id" | "savedAt" | "reviewCount">): Promise<VaultEntry> {
  const id = (globalThis.crypto?.randomUUID?.() ?? `v_${Date.now()}_${Math.random().toString(36).slice(2)}`);
  const full: VaultEntry = {
    ...entry,
    id,
    savedAt: Date.now(),
    reviewCount: 0,
  };
  await set(PREFIX + id, full);
  return full;
}

export async function listEntries(): Promise<VaultEntry[]> {
  const allKeys = (await keys()).filter((k): k is string => typeof k === "string" && k.startsWith(PREFIX));
  const entries = await Promise.all(allKeys.map((k) => get<VaultEntry>(k)));
  return entries
    .filter((e): e is VaultEntry => !!e)
    .sort((a, b) => b.savedAt - a.savedAt);
}

export async function deleteEntry(id: string): Promise<void> {
  await del(PREFIX + id);
}

export async function markReviewed(id: string): Promise<void> {
  const e = await get<VaultEntry>(PREFIX + id);
  if (!e) return;
  e.reviewCount += 1;
  e.lastReviewedAt = Date.now();
  await set(PREFIX + id, e);
}
