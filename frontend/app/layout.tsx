import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Crop Disease AI — Smart Detection & Treatment",
  description:
    "AI-powered plant disease detection system using CNN image classification with generative AI treatment recommendations for farmers.",
  keywords: ["crop disease", "AI", "plant health", "agriculture", "disease detection"],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen flex flex-col">
        {/* Navigation Bar */}
        <nav className="sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b border-gray-200">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              {/* Logo / Brand */}
              <Link href="/" className="flex items-center gap-3 group">
                <div className="w-10 h-10 bg-leaf-600 rounded-xl flex items-center justify-center shadow-md group-hover:bg-leaf-700 transition-colors">
                  <svg
                    className="w-6 h-6 text-white"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                    />
                  </svg>
                </div>
                <div>
                  <h1 className="text-lg font-bold text-gray-900 leading-tight">
                    Crop Disease AI
                  </h1>
                  <p className="text-xs text-gray-500 leading-tight">
                    Smart Detection & Treatment
                  </p>
                </div>
              </Link>

              {/* Navigation Links */}
              <div className="flex items-center gap-2">
                <Link
                  href="/"
                  className="px-4 py-2 text-sm font-medium text-gray-700 rounded-lg hover:bg-leaf-50 hover:text-leaf-700 transition-colors"
                >
                  Diagnose
                </Link>
                <Link
                  href="/history"
                  className="px-4 py-2 text-sm font-medium text-gray-700 rounded-lg hover:bg-leaf-50 hover:text-leaf-700 transition-colors"
                >
                  History
                </Link>
              </div>
            </div>
          </div>
        </nav>

        {/* Main Content */}
        <main className="flex-1">{children}</main>

        {/* Footer */}
        <footer className="bg-white border-t border-gray-200 mt-auto">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
            <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
              <p className="text-sm text-gray-500">
                AI-Based Crop Disease Detection & Smart Treatment Recommendation System
              </p>
              <p className="text-xs text-gray-400">
                Powered by ResNet18 &amp; Generative AI
              </p>
            </div>
          </div>
        </footer>
      </body>
    </html>
  );
}
