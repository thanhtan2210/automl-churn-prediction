import type { Metadata } from "next";
import { Inter } from "next/font/google";
import Link from "next/link";
import { UserRoundSearch } from "lucide-react";

import { Providers } from "@/components/providers";

import "./globals.css";

const inter = Inter({ subsets: ["latin"], display: "swap", variable: "--font-inter" });

export const metadata: Metadata = {
  title: "Churn Predictor",
  description: "Upload a list of telecom subscribers and find out who is likely to cancel — and why.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="min-h-screen bg-muted/40 antialiased">
        <Providers>
          <header className="border-b bg-linear-to-r from-brand-soft via-background to-background">
            <div className="mx-auto flex h-16 max-w-6xl items-center gap-3 px-4 sm:px-6">
              <Link href="/" className="flex items-center gap-2.5">
                <span className="flex size-9 items-center justify-center rounded-xl bg-primary text-primary-foreground shadow-sm">
                  <UserRoundSearch className="size-5" aria-hidden />
                </span>
                <span className="text-lg font-semibold tracking-tight">Churn Predictor</span>
              </Link>
              <span className="hidden text-sm text-muted-foreground md:inline">
                Find out which subscribers are likely to cancel — and why
              </span>
            </div>
          </header>
          <main className="mx-auto max-w-6xl px-4 py-8 sm:px-6">{children}</main>
        </Providers>
      </body>
    </html>
  );
}
