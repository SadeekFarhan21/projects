import type { Metadata, Viewport } from "next";
import { Analytics } from '@vercel/analytics/next';
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL("https://ting.local"),
  title: {
    default: "Tīng — Listen Your Way Into Chinese",
    template: "%s · Tīng",
  },
  description:
    "An audiobook-first Chinese learning app. Voice check-ins, cultural pop-ups, and a smart vocab vault — built for the mobile commute.",
  applicationName: "Tīng",
  manifest: "/manifest.webmanifest",
  icons: { icon: "/favicon.svg", apple: "/icon.svg" },
  appleWebApp: { capable: true, statusBarStyle: "black-translucent", title: "Tīng" },
  openGraph: {
    title: "Tīng — Listen Your Way Into Chinese",
    description:
      "Audiobook learning that actually sticks. Voice check-ins, cultural pop-ups, and a smart vocab vault.",
    type: "website",
  },
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#F7F6F2" },
    { media: "(prefers-color-scheme: dark)", color: "#0F1115" },
  ],
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        {children}
        <Analytics />
        <script
          dangerouslySetInnerHTML={{
            __html: `if ('serviceWorker' in navigator) { window.addEventListener('load', () => { navigator.serviceWorker.register('/sw.js').catch(()=>{}); }); }`,
          }}
        />
      </body>
    </html>
  );
}
