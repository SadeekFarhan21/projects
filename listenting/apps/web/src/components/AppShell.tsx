"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";
import { cn } from "@/lib/cn";
import { Book, Layers, Gear } from "./Icons";

interface Props {
  children: ReactNode;
  hideNav?: boolean;
}

const NAV = [
  { href: "/app", label: "Listen", icon: Book },
  { href: "/app/vault", label: "Vault", icon: Layers },
  { href: "/app/settings", label: "Settings", icon: Gear },
];

export function AppShell({ children, hideNav = false }: Props) {
  const pathname = usePathname();
  return (
    <div className="min-h-dvh flex flex-col bg-ink-50">
      <main className="flex-1 pb-28 pt-safe">{children}</main>
      {!hideNav && (
        <nav className="fixed bottom-0 inset-x-0 z-30 pb-safe">
          <div className="mx-auto max-w-md px-4 pb-3">
            <div className="ink-card flex items-center justify-around p-2">
              {NAV.map((item) => {
                const active =
                  item.href === "/app"
                    ? pathname === "/app" || pathname?.startsWith("/app/listen")
                    : pathname?.startsWith(item.href);
                const Icon = item.icon;
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={cn(
                      "flex flex-col items-center gap-0.5 px-4 py-2 rounded-full transition",
                      active ? "text-ink" : "text-ink-300 hover:text-ink-500",
                    )}
                  >
                    <Icon width={22} height={22} />
                    <span className="text-[10px] uppercase tracking-wider font-medium">{item.label}</span>
                  </Link>
                );
              })}
            </div>
          </div>
        </nav>
      )}
    </div>
  );
}
