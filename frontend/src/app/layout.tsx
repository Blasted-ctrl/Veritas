import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Veritas | AI media detector",
  description:
    "Check images and video frames with an open source AI detector. Inspect the model score, source, and limitations.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
