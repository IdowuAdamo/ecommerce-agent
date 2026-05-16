import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NaijaShop AI — Intelligent Nigerian E-Commerce Assistant",
  description:
    "Multi-agent AI powered shopping assistant for Nigerian e-commerce. Find the best deals on Jumia and Konga with price fairness scoring, trust analysis, and personalized recommendations.",
  keywords: [
    "Nigerian e-commerce", "Jumia AI", "Konga", "product recommendations",
    "price comparison Nigeria", "shopping assistant Nigeria",
  ],
  openGraph: {
    title: "NaijaShop AI",
    description: "Your intelligent Nigerian shopping assistant powered by AI",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased">{children}</body>
    </html>
  );
}
