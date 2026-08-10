import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Bipartite Graph Lab — k/θ-MDBB Visual Debugger",
  description:
    "Visual editor, reduction phase, and k/θ-MDBB search debugger for temporal-mkdb",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="h-full">
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
