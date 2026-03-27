"use client";

import { useState } from "react";
import Upload from "./components/Upload";
import Result from "./components/Result";
import Loader from "./components/Loader";
import { predictDisease, PredictionResult } from "@/lib/api";

export default function HomePage() {
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);

  const handleUpload = async (file: File, treatmentParams?: Record<string, any>) => {
    setLoading(true);
    setError(null);
    setResult(null);

    // Generate image preview
    const previewUrl = URL.createObjectURL(file);
    setImagePreview(previewUrl);

    try {
      const prediction = await predictDisease(file, undefined, treatmentParams);
      setResult(prediction);
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : "An unexpected error occurred.";
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setError(null);
    setLoading(false);
    if (imagePreview) {
      URL.revokeObjectURL(imagePreview);
      setImagePreview(null);
    }
  };

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12">
      {/* Hero Section */}
      <div className="text-center mb-10">
        <h1 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-gray-900 mb-4 text-balance">
          Detect Crop Diseases{" "}
          <span className="text-leaf-600">Instantly</span>
        </h1>
        <p className="text-lg text-gray-600 max-w-2xl mx-auto text-balance">
          Scan a leaf with your camera or upload a photo and our AI will
          identify diseases, assess severity, and provide expert treatment
          recommendations.
        </p>

        {/* Feature Pills */}
        <div className="flex flex-wrap items-center justify-center gap-3 mt-6">
          {[
            "15+ Diseases",
            "Multi-Crop Support",
            "Severity Assessment",
            "AI Recommendations",
          ].map((feature) => (
            <span
              key={feature}
              className="inline-flex items-center px-3 py-1 bg-leaf-50 text-leaf-700 text-sm font-medium rounded-full border border-leaf-200"
            >
              <svg
                className="w-3.5 h-3.5 mr-1.5 text-leaf-500"
                fill="currentColor"
                viewBox="0 0 20 20"
              >
                <path
                  fillRule="evenodd"
                  d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                  clipRule="evenodd"
                />
              </svg>
              {feature}
            </span>
          ))}
        </div>
      </div>

      {/* Upload Section */}
      {!result && !loading && (
        <div className="max-w-2xl mx-auto">
          <Upload onUpload={handleUpload} />
        </div>
      )}

      {/* Loading State */}
      {loading && (
        <div className="max-w-2xl mx-auto">
          <Loader imagePreview={imagePreview} />
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="max-w-2xl mx-auto">
          <div className="card border-red-200 bg-red-50">
            <div className="flex items-start gap-3">
              <div className="flex-shrink-0 w-10 h-10 bg-red-100 rounded-full flex items-center justify-center">
                <svg
                  className="w-5 h-5 text-red-600"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
              </div>
              <div className="flex-1">
                <h3 className="text-sm font-semibold text-red-800">
                  Analysis Failed
                </h3>
                <p className="text-sm text-red-700 mt-1">{error}</p>
              </div>
            </div>
            <button onClick={handleReset} className="btn-primary mt-4 w-full">
              Try Again
            </button>
          </div>
        </div>
      )}

      {/* Results */}
      {result && (
        <div>
          <Result result={result} imagePreview={imagePreview} />
          <div className="flex justify-center mt-8">
            <button onClick={handleReset} className="btn-secondary">
              <svg
                className="w-4 h-4 mr-2"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                />
              </svg>
              Analyze Another Image
            </button>
          </div>
        </div>
      )}

      {/* How It Works Section */}
      {!result && !loading && !error && (
        <div className="mt-16">
          <h2 className="section-title text-center">How It Works</h2>
          <p className="section-subtitle text-center mb-8">
            Three simple steps to diagnose your crops
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[
              {
                step: "1",
                title: "Scan or Upload",
                description:
                  "Use your camera to scan a leaf or upload a photo from your device.",
                icon: (
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"
                  />
                ),
              },
              {
                step: "2",
                title: "AI Analysis",
                description:
                  "Our CNN model processes the image to identify diseases with confidence scoring.",
                icon: (
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                  />
                ),
              },
              {
                step: "3",
                title: "Get Treatment",
                description:
                  "Receive detailed recommendations including causes, treatment, and prevention.",
                icon: (
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                  />
                ),
              },
            ].map((item) => (
              <div key={item.step} className="card-hover text-center">
                <div className="w-14 h-14 bg-leaf-100 rounded-2xl flex items-center justify-center mx-auto mb-4">
                  <svg
                    className="w-7 h-7 text-leaf-600"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    {item.icon}
                  </svg>
                </div>
                <div className="inline-flex items-center justify-center w-6 h-6 bg-leaf-600 text-white text-xs font-bold rounded-full mb-2">
                  {item.step}
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  {item.title}
                </h3>
                <p className="text-sm text-gray-600">{item.description}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
