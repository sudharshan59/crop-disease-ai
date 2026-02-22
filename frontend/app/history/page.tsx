"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { getHistory, HistoryItem, HistoryResponse } from "@/lib/api";
import SeverityBadge from "../components/SeverityBadge";

export default function HistoryPage() {
  const [data, setData] = useState<HistoryResponse | null>(null);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function fetchHistory() {
      setLoading(true);
      setError(null);

      try {
        const res = await getHistory(page, 12);
        if (!cancelled) setData(res);
      } catch {
        if (!cancelled)
          setError(
            "Could not load history. Make sure the backend is running."
          );
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    fetchHistory();
    return () => {
      cancelled = true;
    };
  }, [page]);

  return (
    <div className="max-w-6xl mx-auto px-4 py-10">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Prediction History
        </h1>
        <p className="text-gray-500">
          Browse past diagnoses and treatment recommendations.
        </p>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 rounded-xl p-4 mb-6">
          {error}
        </div>
      )}

      {/* Loading skeleton */}
      {loading && (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <div
              key={i}
              className="card animate-pulse space-y-3"
            >
              <div className="h-4 bg-gray-200 rounded w-3/4" />
              <div className="h-3 bg-gray-200 rounded w-1/2" />
              <div className="h-3 bg-gray-200 rounded w-1/3" />
            </div>
          ))}
        </div>
      )}

      {/* Empty state */}
      {!loading && data && (!data.records || data.records.length === 0) && (
        <div className="card text-center py-16">
          <svg
            className="w-16 h-16 text-gray-300 mx-auto mb-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"
            />
          </svg>
          <h2 className="text-xl font-semibold text-gray-700 mb-2">
            No Predictions Yet
          </h2>
          <p className="text-gray-500 mb-6">
            Upload a leaf image to get started with disease detection.
          </p>
          <Link
            href="/"
            className="btn-primary inline-block"
          >
            Diagnose a Crop
          </Link>
        </div>
      )}

      {/* History cards */}
      {!loading && data && data.records && data.records.length > 0 && (
        <>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {data.records.map((item: HistoryItem) => (
              <HistoryCard key={item.id} item={item} />
            ))}
          </div>

          {/* Pagination */}
          {Math.ceil(data.total / data.per_page) > 1 && (
            <div className="flex items-center justify-center gap-3 mt-8">
              <button
                disabled={page <= 1}
                onClick={() => setPage((p) => p - 1)}
                className="px-4 py-2 rounded-lg text-sm font-medium border border-gray-300 text-gray-700
                  hover:bg-gray-50 disabled:opacity-40 disabled:cursor-not-allowed transition"
              >
                Previous
              </button>
              <span className="text-sm text-gray-600">
                Page {data.page} of {Math.ceil(data.total / data.per_page)}
              </span>
              <button
                disabled={page >= Math.ceil(data.total / data.per_page)}
                onClick={() => setPage((p) => p + 1)}
                className="px-4 py-2 rounded-lg text-sm font-medium border border-gray-300 text-gray-700
                  hover:bg-gray-50 disabled:opacity-40 disabled:cursor-not-allowed transition"
              >
                Next
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function HistoryCard({ item }: { item: HistoryItem }) {
  const ts = new Date(item.timestamp);
  const dateStr = ts.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
  const timeStr = ts.toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  });

  return (
    <div className="card card-hover space-y-3">
      <div className="flex items-start justify-between">
        <div>
          <h3 className="font-semibold text-gray-900">{item.disease}</h3>
          <p className="text-xs text-gray-500 truncate max-w-[180px]">
            {item.filename}
          </p>
        </div>
        <SeverityBadge severity={item.severity} />
      </div>

      <div className="flex items-center gap-4 text-sm text-gray-600">
        <div className="flex items-center gap-1.5">
          <svg
            className="w-4 h-4 text-gray-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
            />
          </svg>
          {(item.confidence * 100).toFixed(1)}%
        </div>
        <div className="flex items-center gap-1.5">
          <svg
            className="w-4 h-4 text-gray-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          {dateStr} {timeStr}
        </div>
      </div>
    </div>
  );
}
