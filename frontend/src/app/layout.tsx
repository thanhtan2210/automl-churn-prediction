import type { Metadata } from "next";
import Link from "next/link";

import { Providers } from "@/components/providers";

import "./globals.css";

export const metadata: Metadata = {
  title: "Churn Predictor",
  description: "Upload a list of telecom subscribers and find out who is likely to cancel — and why.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-muted/30 antialiased">
        <Providers>
          <header className="border-b bg-background">
            <div className="mx-auto flex h-14 max-w-6xl items-center px-4 sm:px-6">
              <Link href="/" className="font-semibold tracking-tight">
                Churn Predictor
              </Link>
              <span className="ml-3 hidden text-sm text-muted-foreground sm:inline">
                Who is likely to cancel — and why
              </span>
            </div>
          </header>
          <main className="mx-auto max-w-6xl px-4 py-8 sm:px-6">{children}</main>
        </Providers>
      </body>
    </html>
  );
}
