"use client";

import { useState } from "react";
import { PredictionResult } from "@/lib/api";
import SeverityBadge from "./SeverityBadge";

interface ResultProps {
  result: PredictionResult;
  imagePreview: string | null;
}

type TabKey = "symptoms" | "cause" | "chemical" | "organic" | "prevention";

export default function Result({ result, imagePreview }: ResultProps) {
  const [activeTab, setActiveTab] = useState<TabKey>("symptoms");
  const confidencePercent = (result.confidence * 100).toFixed(1);
  const isHealthy = result.severity === "healthy";

  const tabContent: Record<TabKey, string> = {
    symptoms: result.recommendation.visible_symptoms || "No symptoms detected.",
    cause: result.recommendation.probable_cause || "Unknown cause.",
    chemical: result.recommendation.treatment?.chemical_control || "No chemical treatment recommended.",
    organic: result.recommendation.treatment?.organic_control || "No organic treatment available.",
    prevention: result.recommendation.prevention || "Follow standard crop management practices.",
  };

  const tabs: { key: TabKey; label: string; icon: React.ReactNode }[] = [
    {
      key: "symptoms",
      label: "Symptoms",
      icon: (
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
        />
      ),
    },
    {
      key: "cause",
      label: "Cause",
      icon: (
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
        />
      ),
    },
    {
      key: "chemical",
      label: "Chemical",
      icon: (
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"
        />
      ),
    },
    {
      key: "organic",
      label: "Organic",
      icon: (
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z"
        />
      ),
    },
    {
      key: "prevention",
      label: "Prevention",
      icon: (
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
        />
      ),
    },
  ];

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Summary Card */}
      <div className="card">
        <div className="flex flex-col sm:flex-row gap-6">
          {/* Image Preview */}
          {imagePreview && (
            <div className="flex-shrink-0">
              <div className="w-full sm:w-48 h-48 rounded-xl overflow-hidden bg-gray-100 border border-gray-200">
                <img
                  src={imagePreview}
                  alt="Analyzed leaf"
                  className="w-full h-full object-cover"
                />
              </div>
            </div>
          )}

          {/* Diagnosis Info */}
          <div className="flex-1 space-y-4">
            <div>
              <div className="flex items-center gap-3 mb-2">
                <h2 className="text-2xl font-bold text-gray-900">
                  {result.recommendation.disease_name || result.disease}
                </h2>
                <SeverityBadge severity={result.severity} />
              </div>

              {result.recommendation.crop_name && (
                <p className="text-sm font-medium text-leaf-600 mb-1">
                  Crop: {result.recommendation.crop_name}
                </p>
              )}

              <p className="text-sm text-gray-500">
                {isHealthy
                  ? "No disease detected. Your plant appears healthy!"
                  : "Disease detected. See recommendations below."}
              </p>

              {result.recommendation.confidence && (
                <p className="text-xs text-gray-400 mt-1">
                  VLM Diagnostic Confidence: {result.recommendation.confidence}
                </p>
              )}
            </div>

            {/* Confidence Bar */}
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-sm font-medium text-gray-700">
                  Confidence
                </span>
                <span className="text-sm font-bold text-gray-900">
                  {confidencePercent}%
                </span>
              </div>
              <div className="w-full h-3 bg-gray-200 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-1000 ease-out ${
                    isHealthy
                      ? "bg-green-500"
                      : result.confidence >= 0.9
                      ? "bg-red-500"
                      : result.confidence >= 0.7
                      ? "bg-orange-500"
                      : "bg-yellow-500"
                  }`}
                  style={{ width: `${confidencePercent}%` }}
                />
              </div>
            </div>

            {/* Quick Stats */}
            <div className="grid grid-cols-3 gap-3">
              <div className="bg-gray-50 rounded-xl p-3 text-center">
                <p className="text-xs text-gray-500 mb-1">Status</p>
                <p className="text-sm font-semibold text-gray-900">
                  {isHealthy ? "Healthy" : "Diseased"}
                </p>
              </div>
              <div className="bg-gray-50 rounded-xl p-3 text-center">
                <p className="text-xs text-gray-500 mb-1">Severity</p>
                <p className="text-sm font-semibold text-gray-900 capitalize">
                  {result.severity}
                </p>
              </div>
              <div className="bg-gray-50 rounded-xl p-3 text-center">
                <p className="text-xs text-gray-500 mb-1">Confidence</p>
                <p className="text-sm font-semibold text-gray-900">
                  {confidencePercent}%
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Recommendations Card */}
      <div className="card">
        <h3 className="section-title flex items-center gap-2 mb-1">
          <svg
            className="w-6 h-6 text-leaf-600"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"
            />
          </svg>
          AI Recommendations
        </h3>
        <p className="section-subtitle mb-6">
          Expert-level treatment guidance generated by AI
        </p>

        {/* Tabs */}
        <div className="flex gap-1 bg-gray-100 rounded-xl p-1 mb-6 overflow-x-auto">
          {tabs.map((tab) => (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              className={`
                flex items-center gap-1.5 px-4 py-2.5 rounded-lg text-sm font-medium
                transition-all duration-200 whitespace-nowrap flex-1
                ${
                  activeTab === tab.key
                    ? "bg-white text-leaf-700 shadow-sm"
                    : "text-gray-600 hover:text-gray-900"
                }
              `}
            >
              <svg
                className="w-4 h-4 hidden sm:block"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                {tab.icon}
              </svg>
              {tab.label}
            </button>
          ))}
        </div>

        {/* Tab Content */}
        <div className="bg-gray-50 rounded-xl p-5 sm:p-6 min-h-[160px]">
          <p className="text-gray-700 leading-relaxed whitespace-pre-line">
            {tabContent[activeTab]}
          </p>
        </div>
      </div>
    </div>
  );
}
