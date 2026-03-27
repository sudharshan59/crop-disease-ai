"""
Configuration module for the Crop Disease AI system.
Centralizes all settings using Pydantic BaseSettings with .env support.
"""

import os
import torch
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


# ─── Base Directories ─────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DB_DIR = BASE_DIR / "data"
CONTEXT_DIR = BASE_DIR / "context"


# ─── Disease Class Labels (16 classes: 15 diseases + 1 healthy) ───────────────
DISEASE_CLASSES = [
    "Apple___Apple_scab",
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot",
    "Corn_(maize)___Common_rust",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Target_Spot",
    "Tomato___Yellow_Leaf_Curl_Virus",
    "Healthy",
]

NUM_CLASSES = len(DISEASE_CLASSES)

# Human-readable disease names for display
DISEASE_DISPLAY_NAMES = {
    "Apple___Apple_scab": "Apple Scab",
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot": "Corn Gray Leaf Spot",
    "Corn_(maize)___Common_rust": "Corn Common Rust",
    "Corn_(maize)___Northern_Leaf_Blight": "Corn Northern Leaf Blight",
    "Grape___Black_rot": "Grape Black Rot",
    "Grape___Esca_(Black_Measles)": "Grape Black Measles (Esca)",
    "Potato___Early_blight": "Potato Early Blight",
    "Potato___Late_blight": "Potato Late Blight",
    "Tomato___Bacterial_spot": "Tomato Bacterial Spot",
    "Tomato___Early_blight": "Tomato Early Blight",
    "Tomato___Late_blight": "Tomato Late Blight",
    "Tomato___Leaf_Mold": "Tomato Leaf Mold",
    "Tomato___Septoria_leaf_spot": "Tomato Septoria Leaf Spot",
    "Tomato___Target_Spot": "Tomato Target Spot",
    "Tomato___Yellow_Leaf_Curl_Virus": "Tomato Yellow Leaf Curl Virus",
    "Healthy": "Healthy",
}


# ─── ImageNet Normalization Constants ─────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ─── Image Preprocessing Constants ───────────────────────────────────────────
IMAGE_SIZE = 224
GAUSSIAN_BLUR_KERNEL = (5, 5)


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    # ── API Settings ──────────────────────────────────────────────────────
    API_HOST: str = Field(default="0.0.0.0", description="API server host")
    API_PORT: int = Field(default=8000, description="API server port")
    API_RELOAD: bool = Field(default=False, description="Auto-reload on code changes")
    CORS_ORIGINS: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000",
        description="Comma-separated list of allowed CORS origins",
    )

    # ── Model Settings ────────────────────────────────────────────────────
    MODEL_PATH: str = Field(
        default=str(MODELS_DIR / "plant_disease_model.pth"),
        description="Path to trained CNN model weights (.pth or .pt)",
    )
    TORCHSCRIPT_PATH: str = Field(
        default=str(MODELS_DIR / "plant_disease_model_scripted.pt"),
        description="Path to TorchScript exported model",
    )
    DEVICE: str = Field(
        default="auto",
        description="Device for inference: 'auto', 'cpu', or 'cuda'",
    )

    # ── VLM Settings (Vision-Language Model) ────────────────────────────
    LLM_BACKEND: str = Field(
        default="gguf",
        description="LLM backend: 'gguf' (llama-cpp-python) or 'bitsandbytes'",
    )
    LLM_MODEL_NAME: str = Field(
        default="xtuner/llava-phi-3-mini-gguf",
        description="HuggingFace model name / repo",
    )
    LLM_GGUF_PATH: str = Field(
        default=str(MODELS_DIR / "llava-phi-3-mini-int4.gguf"),
        description="Path to INT4 quantized LLaVA-Phi-3-Mini GGUF (~2.3GB)",
    )
    LLM_MMPROJ_PATH: str = Field(
        default=str(MODELS_DIR / "llava-phi-3-mini-mmproj-f16.gguf"),
        description="Path to CLIP vision projection GGUF for multimodal support",
    )
    LLM_MAX_TOKENS: int = Field(
        default=400,
        description="Maximum tokens for LLM generation (treatment text only)",
    )
    LLM_TEMPERATURE: float = Field(
        default=0.0,
        description="Temperature for VLM generation (0.0-1.0, higher = more creative)",
    )
    LLM_CONTEXT_LENGTH: int = Field(
        default=1024,
        description="Context window size for GGUF model (smaller = faster)",
    )
    VLM_TIMEOUT: int = Field(
        default=60,
        description="Max seconds to wait for VLM inference before falling back to dictionary",
    )
    VLM_ENABLED: bool = Field(
        default=True,
        description="Enable VLM image analysis (set False to use fast fallback dictionary only)",
    )
    VLM_IMAGE_MAX_SIZE: int = Field(
        default=224,
        description="Resize uploaded images to NxN before sending to VLM (smaller = faster on CPU)",
    )

    # ── Database Settings ─────────────────────────────────────────────────
    DB_PATH: str = Field(
        default=str(DB_DIR / "predictions.db"),
        description="Path to SQLite database file",
    )

    # ── Logging ───────────────────────────────────────────────────────────
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_PROMPTS: bool = Field(
        default=True,
        description="Log LLM prompts for refinement analysis",
    )

    class Config:
        env_file = str(BASE_DIR / ".env")
        env_file_encoding = "utf-8"
        case_sensitive = True

    def get_device(self) -> str:
        """Resolve the compute device."""
        if self.DEVICE == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.DEVICE

    def get_cors_origins(self) -> list[str]:
        """Parse comma-separated CORS origins into a list."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]


# ─── Global Settings Instance ────────────────────────────────────────────────
settings = Settings()
