"use client";

import { ThemeProvider as NextThemesProvider } from "next-themes";
import type { ComponentProps } from "react";

/**
 * Thin wrapper around next-themes, added by the shadcn `sonner` install
 * (its Toaster reads the current theme via next-themes' useTheme()).
 * `attribute="class"` toggles the `.dark` class shadcn's globals.css
 * already defines variants for — nothing else in the app reads theme
 * state yet, this just wires system-preference dark mode up correctly
 * instead of leaving `useTheme()` with no provider above it.
 */
export function ThemeProvider({ children, ...props }: ComponentProps<typeof NextThemesProvider>) {
  return <NextThemesProvider {...props}>{children}</NextThemesProvider>;
}
