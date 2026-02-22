"""
Pydantic schemas for API request/response validation.

Defines structured models for the prediction endpoint responses,
history records, and health checks.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class TreatmentSchema(BaseModel):
    """Treatment recommendation split into chemical and organic methods."""

    chemical_control: str = Field(
        ...,
        description="Chemical/fungicide treatment recommendations",
        examples=["Apply mancozeb or chlorothalonil fungicide at 7-10 day intervals."],
    )
    organic_control: str = Field(
        ...,
        description="Organic or biological control recommendations",
        examples=["Apply Bacillus subtilis biofungicide. Use neem oil spray."],
    )


class RecommendationSchema(BaseModel):
    """Structured diagnostic report from the VLM / fallback module."""

    crop_name: str = Field(
        ...,
        description="Identified crop type",
        examples=["Tomato"],
    )
    disease_name: str = Field(
        ...,
        description="Most likely disease name",
        examples=["Early Blight"],
    )
    visible_symptoms: str = Field(
        ...,
        description="Observable symptoms described clearly",
        examples=["Dark brown lesions with concentric rings on lower leaves."],
    )
    probable_cause: str = Field(
        ...,
        description="Probable cause including pathogen type and conditions",
        examples=["Fungal infection by Alternaria solani, favored by warm humid conditions."],
    )
    treatment: TreatmentSchema = Field(
        ...,
        description="Treatment recommendations split into chemical and organic methods",
    )
    prevention: str = Field(
        ...,
        description="Prevention and farm management practices",
        examples=["Practice crop rotation. Use resistant varieties. Mulch plants."],
    )
    confidence: str = Field(
        default="Medium",
        description="Diagnostic confidence level: Low / Medium / High",
        examples=["High"],
    )


class PredictionResponse(BaseModel):
    """Response schema for the /api/predict endpoint."""

    disease: str = Field(
        ...,
        description="Human-readable disease name",
        examples=["Tomato Early Blight"],
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence score (0.0 to 1.0)",
        examples=[0.9523],
    )
    severity: str = Field(
        ...,
        description="Disease severity level: healthy, mild, moderate, or severe",
        examples=["moderate"],
    )
    recommendation: RecommendationSchema = Field(
        ...,
        description="AI-generated treatment recommendations",
    )


class HistoryItem(BaseModel):
    """Schema for a single prediction history record."""

    id: int = Field(..., description="Record ID")
    timestamp: str = Field(
        ...,
        description="ISO 8601 timestamp of the prediction",
        examples=["2026-02-22T14:30:00"],
    )
    filename: str = Field(
        ...,
        description="Original uploaded image filename",
        examples=["tomato_leaf.jpg"],
    )
    disease: str = Field(
        ...,
        description="Detected disease name",
        examples=["Tomato Early Blight"],
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence score",
    )
    severity: str = Field(
        ...,
        description="Severity level",
        examples=["moderate"],
    )
    recommendation: RecommendationSchema = Field(
        ...,
        description="Treatment recommendations",
    )
    region: Optional[str] = Field(
        default=None,
        description="Optional region for context-aware recommendations",
        examples=["Southeast Asia"],
    )


class HistoryResponse(BaseModel):
    """Response schema for the /api/history endpoint."""

    total: int = Field(..., description="Total number of prediction records")
    page: int = Field(..., description="Current page number (1-indexed)")
    per_page: int = Field(..., description="Number of records per page")
    records: List[HistoryItem] = Field(
        default_factory=list,
        description="List of prediction history records",
    )


class HealthResponse(BaseModel):
    """Response schema for the /api/health endpoint."""

    status: str = Field(
        ...,
        description="API health status",
        examples=["ok"],
    )
    cnn_model: dict = Field(
        ...,
        description="CNN model status information",
    )
    llm_model: dict = Field(
        ...,
        description="LLM model status information",
    )
    device: str = Field(
        ...,
        description="Compute device in use",
        examples=["cpu"],
    )
