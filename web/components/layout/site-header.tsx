"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { LogoMark } from "@/components/ui/logo";
import { cn } from "@/lib/utils";

const nav = [
  { href: "/chat", label: "Ask" },
  { href: "/notebook", label: "Notebook" },
  { href: "/explorer", label: "Explorer" },
  { href: "/datasets", label: "Datasets" },
];

export function SiteHeader() {
  const pathname = usePathname() ?? "/";

  return (
    <header className="sticky top-0 z-30 border-b border-hairline bg-canvas/85 backdrop-blur">
      <div className="mx-auto flex h-16 max-w-[1400px] items-center gap-6 px-4 md:px-6 lg:px-8">
        <Link
          href="/"
          className="flex items-center gap-2.5 rounded-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy"
          aria-label="MapleQuery home"
        >
          <LogoMark />
          <span className="text-lg font-semibold tracking-tight text-ink">
            MapleQuery
          </span>
        </Link>

        <nav
          className="ml-auto hidden items-center gap-1 md:flex"
          aria-label="Primary"
        >
          {nav.map((item) => {
            const active =
              pathname === item.href ||
              (item.href !== "/" && pathname.startsWith(item.href));
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "rounded-md px-3 py-1.5 text-sm font-medium transition-colors",
                  active
                    ? "bg-surface-soft text-ink"
                    : "text-muted hover:bg-surface-soft/60 hover:text-ink",
                )}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
