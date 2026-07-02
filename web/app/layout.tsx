import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "@/styles/globals.css";
import { SiteHeader } from "@/components/layout/site-header";
import { ToastProvider } from "@/components/ui/toast";
import { PostHogProvider } from "@/components/analytics/posthog-provider";

const inter = Inter({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-sans",
  display: "swap",
});

export const metadata: Metadata = {
  title: "MapleQuery · Ask hard questions of Canadian government data",
  description:
    "MapleQuery turns fragmented Canadian open data into a plain-language conversation. Every figure carries a footnote that traces to the original record.",
  metadataBase: new URL("https://maple-query.vercel.app"),
  icons: {
    icon: [{ url: "/brand/maple-leaf.webp", type: "image/webp" }],
    shortcut: "/brand/maple-leaf.webp",
    apple: "/brand/maple-leaf.webp",
  },
  openGraph: {
    title: "MapleQuery",
    description:
      "Ask hard questions of Canadian government data. Get answers you can cite.",
    type: "website",
    images: ["/brand/maple-leaf.webp"],
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="min-h-screen bg-canvas font-sans text-body antialiased">
        <div aria-hidden="true" className="site-backdrop" />
        <PostHogProvider>
          <ToastProvider>
            <div className="relative flex min-h-screen flex-col">
              <SiteHeader />
              <div className="flex-1">{children}</div>
            </div>
          </ToastProvider>
        </PostHogProvider>
      </body>
    </html>
  );
}
