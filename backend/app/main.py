"""
FastAPI Application — Crop Disease Detection & Treatment Recommendation System.

Provides endpoints for:
- Health check with model status
- Image-based disease prediction with AI recommendations
- Prediction history retrieval

Uses lifespan context manager for model initialization and cleanup.
"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.model import classifier
from app.llm import recommendation_engine
from app.preprocessing import preprocess_image
from app.severity import estimate_severity
from app.database import init_db, save_prediction, get_history
from app.schemas import (
    PredictionResponse,
    RecommendationSchema,
    TreatmentSchema,
    HistoryResponse,
    HealthResponse,
)

# ─── Logging Setup ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ─── Lifespan — Model Loading / Cleanup ──────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load models on startup, clean up on shutdown."""
    logger.info("=" * 60)
    logger.info("  Crop Disease AI — Starting Up")
    logger.info("=" * 60)

    # Initialize database
    await init_db()
    logger.info("Database initialized.")

    # Load CNN disease classifier
    logger.info("Loading CNN disease classifier...")
    cnn_loaded = classifier.load()
    if cnn_loaded:
        logger.info("CNN model loaded successfully.")
    else:
        logger.warning("CNN model failed to load. Predictions will not be available.")

    # Load LLM recommendation engine
    logger.info("Loading LLM recommendation engine...")
    llm_loaded = recommendation_engine.load()
    if llm_loaded:
        logger.info("LLM loaded successfully.")
    else:
        logger.info("LLM not available. Using fallback recommendations.")

    logger.info("=" * 60)
    logger.info("  System Ready")
    logger.info(f"  Device: {settings.get_device()}")
    logger.info(f"  CNN: {'loaded' if cnn_loaded else 'not loaded'}")
    logger.info(f"  LLM: {recommendation_engine.get_status()['backend']}")
    logger.info("=" * 60)

    yield  # Application runs

    # Shutdown cleanup
    logger.info("Shutting down Crop Disease AI...")
    # Free GPU memory if used
    if classifier.model is not None:
        del classifier.model
    if recommendation_engine.model is not None:
        del recommendation_engine.model
    logger.info("Cleanup complete. Goodbye.")


# ─── FastAPI App ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Crop Disease Detection & Treatment Recommendation API",
    description=(
        "AI-powered plant disease detection using CNN image classification "
        "with generative AI treatment recommendations. Upload a leaf image "
        "to receive disease diagnosis, severity assessment, and actionable "
        "treatment guidance."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ─── CORS Middleware ─────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════


@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check system health status and model availability.

    Returns status of the CNN classifier, LLM engine, and compute device.
    """
    return HealthResponse(
        status="ok",
        cnn_model=classifier.get_status(),
        llm_model=recommendation_engine.get_status(),
        device=settings.get_device(),
    )


@app.post("/api/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_disease(
    file: UploadFile = File(..., description="Leaf image file (JPEG, PNG)"),
    region: Optional[str] = Query(
        default=None,
        description="Optional geographic region for context-aware recommendations",
        examples=["South Asia", "Southeast Asia", "Sub-Saharan Africa"],
    ),
    treatment_params: Optional[str] = Form(
        default=None,
        description="Optional JSON string with treatment adjustment parameters (chemical_amount_g, water_amount_l, organic_amount_g, notes)",
    ),
):
    """Analyze a leaf image for disease detection and generate treatment recommendations.

    Pipeline:
    1. Validate and read uploaded image
    2. Preprocess with OpenCV (resize, blur, normalize, optional segmentation)
    3. Run CNN inference for disease classification
    4. Estimate disease severity
    5. Generate AI-powered treatment recommendations
    6. Save prediction to history database

    Args:
        file: Uploaded leaf image file.
        region: Optional region for localized recommendations.

    Returns:
        PredictionResponse with disease, confidence, severity, and recommendations.

    Raises:
        HTTPException 400: Invalid image file.
        HTTPException 503: CNN model not loaded.
    """
    # ── Validate file type ────────────────────────────────────────────────
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Please upload an image (JPEG, PNG).",
        )

    # ── Read image bytes ──────────────────────────────────────────────────
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {str(e)}")

    # ── Check model availability ──────────────────────────────────────────
    if not classifier.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="CNN model is not loaded. Please check server logs.",
        )

    # (image_bytes already read above)

    # Parse treatment params (if provided)
    tparams = None
    if treatment_params:
        try:
            tparams = __import__("json").loads(treatment_params)
        except Exception:
            tparams = None

    # ── Run inference pipeline in executor (avoid blocking event loop) ────
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: _run_prediction_pipeline(image_bytes, region, tparams),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal prediction error. Check server logs.")

    # ── Save to database ──────────────────────────────────────────────────
    try:
        await save_prediction(
            filename=file.filename or "unknown.jpg",
            disease=result["disease"],
            confidence=result["confidence"],
            severity=result["severity"],
            recommendation=result["recommendation"],
            region=region,
        )
    except Exception as e:
        logger.error(f"Failed to save prediction to database: {e}", exc_info=True)
        # Non-critical — still return the prediction to the user

    # ── Build response ────────────────────────────────────────────────────
    rec = result["recommendation"]
    # Normalize treatment: if it's a dict with chemical/organic, wrap it
    treatment_data = rec.get("treatment", {})
    if isinstance(treatment_data, dict):
        treatment_obj = TreatmentSchema(**treatment_data)
    else:
        treatment_obj = TreatmentSchema(
            chemical_control=str(treatment_data),
            organic_control="",
        )

    return PredictionResponse(
        disease=result["disease"],
        confidence=result["confidence"],
        severity=result["severity"],
        recommendation=RecommendationSchema(
            crop_name=rec.get("crop_name", "Unknown"),
            disease_name=rec.get("disease_name", result["disease"]),
            visible_symptoms=rec.get("visible_symptoms", ""),
            probable_cause=rec.get("probable_cause", ""),
            treatment=treatment_obj,
            prevention=rec.get("prevention", ""),
            confidence=rec.get("confidence", "Medium"),
            sections=rec.get("sections", []),
            weekly_plan=rec.get("weekly_plan", []),
        ),
    )


def _run_prediction_pipeline(image_bytes: bytes, region: Optional[str] = None, treatment_params: Optional[dict] = None) -> dict:
    """Execute the full prediction pipeline (blocking, runs in executor).

    Args:
        image_bytes: Raw image file bytes.
        region: Optional region string.

    Returns:
        Dictionary with disease, confidence, severity, and recommendation.
    """
    # Step 1: Preprocess image
    tensor, diseased_area_ratio = preprocess_image(image_bytes, apply_segmentation=True)

    # Step 2: CNN inference (run primary and optional alternate, choose winner)
    dual = dual_result = None
    try:
        from app import model as model_module
        dual = model_module.dual_predict(tensor)
    except Exception:
        # fallback to single prediction
        class_idx, class_label, display_name, confidence, probabilities = classifier.predict(tensor)
        dual = {
            "primary": {
                "class_index": class_idx,
                "class_label": class_label,
                "display_name": display_name,
                "confidence": confidence,
                "probabilities": probabilities,
            },
            "alternate": None,
            "winner": {
                "class_index": class_idx,
                "class_label": class_label,
                "display_name": display_name,
                "confidence": confidence,
            },
            "loser": None,
        }

    # Step 3: Severity estimation
    # Ensure we have a valid winner; fall back to single predict if not
    if not dual or "winner" not in dual or dual.get("winner") is None:
        class_idx, class_label, display_name, confidence, probabilities = classifier.predict(tensor)
        disease_name = display_name
    else:
        winner = dual.get("winner")
        disease_name = winner.get("display_name") or winner.get("disease_name")
        confidence = winner.get("confidence")
        class_label = winner.get("class_label")
    is_healthy = classifier.is_healthy(class_label)
    severity = estimate_severity(confidence, is_healthy, diseased_area_ratio)

    # Step 4: Generate recommendation (VLM analyzes image directly)
    recommendation = recommendation_engine.generate_recommendation(
        disease=disease_name,
        confidence=confidence,
        severity=severity,
        region=region,
        image_bytes=image_bytes,
        treatment_params=treatment_params,
    )

    # Log end of pipeline for traceability
    try:
        logger.info(
            "Prediction complete: disease=%s confidence=%.3f severity=%s",
            disease_name,
            confidence,
            severity,
        )
    except Exception:
        pass

    return {
        "disease": disease_name,
        "confidence": confidence,
        "severity": severity,
        "recommendation": recommendation,
    }


@app.get("/api/history", response_model=HistoryResponse, tags=["History"])
async def get_prediction_history(
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    per_page: int = Query(default=20, ge=1, le=100, description="Records per page"),
):
    """Retrieve prediction history with pagination.

    Args:
        page: Page number (1-indexed, default 1).
        per_page: Number of records per page (default 20, max 100).

    Returns:
        HistoryResponse with total count and paginated records.
    """
    try:
        history_data = await get_history(page=page, per_page=per_page)
        return HistoryResponse(**history_data)
    except Exception as e:
        logger.error(f"Failed to retrieve history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve prediction history.")
