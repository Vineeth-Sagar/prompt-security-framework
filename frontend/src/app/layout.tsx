import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

import { AuthProvider } from "@/components/auth-provider";
import { NavBar } from "@/components/nav-bar";
import { ThemeProvider } from "@/components/theme-provider";
import { Toaster } from "@/components/ui/sonner";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Prompt Security Framework",
  description:
    "Dashboard for the Adaptive Context-Aware Multi-Layer Prompt Security & Output Governance Framework.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
      // next-themes sets the .dark class on <html> from a blocking
      // inline script before hydration, based on the browser's actual
      // preference/localStorage — that will legitimately differ from
      // what the server rendered (which has no way to know either).
      // This is next-themes' own documented use of
      // suppressHydrationWarning, scoped to just this one attribute.
      suppressHydrationWarning
    >
      <body className="min-h-full flex flex-col">
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
          <AuthProvider>
            <NavBar />
            <main className="flex flex-1 flex-col">{children}</main>
            <Toaster />
          </AuthProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
