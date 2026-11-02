import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Veritas | AI Image & Video Detector",
  description:
    "Check images and video frames for signs of AI generation. Upload a file, run the analysis, and see the result and AI score with Veritas.",
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
