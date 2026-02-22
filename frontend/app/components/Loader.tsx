"use client";

import { useEffect, useState } from "react";

export default function Loader({
  imagePreview,
}: {
  imagePreview: string | null;
}) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => setElapsed((e) => e + 1), 1000);
    return () => clearInterval(timer);
  }, []);

  const stepStatus = (threshold: number) =>
    elapsed >= threshold ? "text-leaf-600 font-medium" : "text-gray-400";

  const stepDot = (threshold: number) =>
    elapsed >= threshold ? "bg-leaf-500" : "bg-gray-300";

  return (
    <div className="max-w-2xl mx-auto">
      <div className="card text-center space-y-6 py-10">
        {/* Image Preview Skeleton */}
        {imagePreview && (
          <div className="mx-auto w-40 h-40 rounded-xl overflow-hidden border-2 border-leaf-200 shadow-md relative">
            <img
              src={imagePreview}
              alt="Analyzing..."
              className="w-full h-full object-cover opacity-50"
            />
            <div className="absolute inset-0 bg-gradient-to-t from-leaf-500/30 to-transparent animate-pulse" />
          </div>
        )}

        {/* Spinner */}
        <div className="flex items-center justify-center">
          <div className="relative">
            <div className="w-16 h-16 border-4 border-leaf-200 rounded-full" />
            <div className="w-16 h-16 border-4 border-transparent border-t-leaf-600 rounded-full absolute inset-0 animate-spin" />
            <svg
              className="w-7 h-7 text-leaf-600 absolute inset-0 m-auto"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
              />
            </svg>
          </div>
        </div>

        {/* Status Text */}
        <div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">
            Analyzing Your Crop Image
          </h3>
          <p className="text-gray-500 text-sm max-w-sm mx-auto">
            {elapsed < 3
              ? "Preprocessing and classifying the image..."
              : elapsed < 10
              ? "AI model is generating treatment recommendations..."
              : "AI is analyzing disease patterns and medicines — almost done..."}
          </p>
          <p className="text-xs text-gray-400 mt-2">
            Elapsed: {elapsed}s
          </p>
        </div>

        {/* Progress Steps */}
        <div className="flex flex-col items-start max-w-xs mx-auto space-y-3 text-sm">
          {[
            { label: "Preprocessing image", threshold: 0 },
            { label: "Running CNN disease classification", threshold: 2 },
            { label: "Estimating severity level", threshold: 3 },
            { label: "AI generating treatment medicines", threshold: 5 },
          ].map((step, i) => (
            <div key={i} className="flex items-center gap-3">
              <div className={`w-2 h-2 rounded-full ${stepDot(step.threshold)} ${elapsed >= step.threshold && elapsed < step.threshold + 60 ? "animate-pulse" : ""}`} />
              <span className={stepStatus(step.threshold)}>{step.label}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
