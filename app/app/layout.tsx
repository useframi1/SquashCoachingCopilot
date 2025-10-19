import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Squash Coaching Copilot",
  description: "AI-powered squash match analysis and coaching insights",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
