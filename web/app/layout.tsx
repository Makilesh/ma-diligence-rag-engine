import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import { BRAND } from "@/lib/brand";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: `${BRAND.name} — ${BRAND.tagline}`,
  description: BRAND.promise,
  applicationName: BRAND.name,
};

export const viewport: Viewport = {
  themeColor: "#06080c",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="antialiased">
        {/* Fixed ambient layers. Kept out of the scrolling tree so they do not
            repaint on every scroll frame. */}
        <div className="canvas-glow" aria-hidden />
        <div className="canvas-grid" aria-hidden />
        <div className="canvas-grain" aria-hidden />
        <div className="relative z-10">{children}</div>
      </body>
    </html>
  );
}
