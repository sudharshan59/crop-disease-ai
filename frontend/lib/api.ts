import axios from "axios";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const api = axios.create({
  baseURL: API_URL,
  timeout: 300_000, // 5 min — LLM on CPU can be slow
});

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

export interface Treatment {
  chemical_control: string;
  organic_control: string;
}

export interface Recommendation {
  crop_name: string;
  disease_name: string;
  visible_symptoms: string;
  probable_cause: string;
  treatment: Treatment;
  prevention: string;
  confidence: string;
}

export interface PredictionResult {
  disease: string;
  confidence: number;
  severity: string;
  recommendation: Recommendation;
  timestamp?: string;
  filename?: string;
}

export interface HistoryItem {
  id: number;
  timestamp: string;
  filename: string;
  disease: string;
  confidence: number;
  severity: string;
}

export interface HistoryResponse {
  records: HistoryItem[];
  total: number;
  page: number;
  per_page: number;
}

export interface HealthStatus {
  status: string;
  model_loaded: boolean;
  llm_loaded: boolean;
  device: string;
}

/* ------------------------------------------------------------------ */
/*  API Functions                                                      */
/* ------------------------------------------------------------------ */

/**
 * Send a leaf image to the backend for disease prediction.
 */
export async function predictDisease(
  file: File,
  region?: string
): Promise<PredictionResult> {
  const formData = new FormData();
  formData.append("file", file);
  if (region) formData.append("region", region);

  const { data } = await api.post<PredictionResult>(
    "/api/predict",
    formData,
    {
      headers: { "Content-Type": "multipart/form-data" },
    }
  );
  return data;
}

/**
 * Fetch prediction history with pagination.
 */
export async function getHistory(
  page: number = 1,
  perPage: number = 10
): Promise<HistoryResponse> {
  const { data } = await api.get<HistoryResponse>("/api/history", {
    params: { page, per_page: perPage },
  });
  return data;
}

/**
 * Check if the backend is healthy.
 */
export async function healthCheck(): Promise<HealthStatus> {
  const { data } = await api.get<HealthStatus>("/api/health");
  return data;
}

export default api;
