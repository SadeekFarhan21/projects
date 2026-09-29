import { notFound } from "next/navigation";
import { Player } from "@/components/Player";
import { getChapter } from "@/lib/data";

export const dynamic = "force-dynamic";

export default async function ListenPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const chapter = await getChapter(id);
  if (!chapter) notFound();
  return <Player chapter={chapter} />;
}
