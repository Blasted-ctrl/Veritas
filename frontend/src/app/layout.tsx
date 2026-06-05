import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Veritas — Deepfake & AI-Content Detector",
  description:
    "Upload an image, audio clip, or video for an authenticity verdict with confidence breakdown and a Grad-CAM interpretability heatmap.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-ink bg-gradient-to-b from-ink to-[#0a0f1d] text-slate-100 antialiased">
        {children}
      </body>
    </html>
  );
}
