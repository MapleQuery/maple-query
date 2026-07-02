import type { Metadata } from "next";
import { Fraunces, Inter, Fira_Code } from "next/font/google";
import "@/styles/globals.css";
import { cn } from "@/lib/utils";
import { SiteHeader } from "@/components/layout/site-header";
import { ToastProvider } from "@/components/ui/toast";

const inter = Inter({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-sans",
  display: "swap",
});

const fraunces = Fraunces({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-display",
  display: "swap",
});

const firaCode = Fira_Code({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "MapleQuery — Ask hard questions of Canadian government data",
  description:
    "MapleQuery turns fragmented Canadian open data into a plain-language conversation. Every figure carries a footnote that traces to the original record.",
  metadataBase: new URL("https://maplequery.vercel.app"),
  openGraph: {
    title: "MapleQuery",
    description:
      "Ask hard questions of Canadian government data — and get answers you can cite.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="en"
      className={cn(inter.variable, fraunces.variable, firaCode.variable)}
    >
      <body className="min-h-screen bg-canvas font-sans text-body antialiased">
        <ToastProvider>
          <div className="flex min-h-screen flex-col">
            <SiteHeader />
            <div className="flex-1">{children}</div>
          </div>
        </ToastProvider>
      </body>
    </html>
  );
}
