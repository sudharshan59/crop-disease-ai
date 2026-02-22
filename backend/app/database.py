"""
SQLite database module for prediction history storage.

Uses aiosqlite for async database operations to avoid blocking
the FastAPI event loop.
"""

import json
import logging
import aiosqlite
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)

# ─── Database Path ────────────────────────────────────────────────────────────
DB_PATH = settings.DB_PATH


def _normalize_recommendation(rec: dict, disease: str = "") -> dict:
    """Ensure a recommendation dict matches the current RecommendationSchema.

    Old records may have keys like 'description', 'causes', 'treatment' (str).
    New schema expects: crop_name, disease_name, visible_symptoms,
    probable_cause, treatment (dict with chemical_control & organic_control),
    prevention, confidence.
    """
    # If already in new format, return as-is
    if "crop_name" in rec and "disease_name" in rec and isinstance(rec.get("treatment"), dict):
        return rec

    # --- Extract crop_name and disease_name from the disease string ---
    crop_name = ""
    disease_name = disease or ""
    if disease:
        parts = disease.split(" ", 1)
        if len(parts) == 2:
            crop_name = parts[0]
            disease_name = parts[1]
        else:
            crop_name = disease
            disease_name = disease

    # --- Map old fields to new fields ---
    visible_symptoms = (
        rec.get("visible_symptoms")
        or rec.get("description")
        or "See uploaded image for symptoms."
    )
    probable_cause = (
        rec.get("probable_cause")
        or rec.get("causes")
        or rec.get("cause")
        or "Refer to disease documentation."
    )
    prevention = rec.get("prevention", "Practice crop rotation and good sanitation.")

    # --- Normalize treatment ---
    raw_treatment = rec.get("treatment", "")
    if isinstance(raw_treatment, dict):
        treatment = {
            "chemical_control": raw_treatment.get("chemical_control", "Consult local extension service."),
            "organic_control": raw_treatment.get("organic_control", "Use organic-approved methods."),
        }
    elif isinstance(raw_treatment, str) and raw_treatment:
        # Old format: treatment was a single string. Split heuristically.
        treatment = {
            "chemical_control": raw_treatment,
            "organic_control": "Consider organic alternatives; consult local extension service.",
        }
    else:
        treatment = {
            "chemical_control": "Consult local extension service.",
            "organic_control": "Use organic-approved methods.",
        }

    confidence = rec.get("confidence", "Medium")
    if isinstance(confidence, (int, float)):
        if confidence > 0.75:
            confidence = "High"
        elif confidence > 0.4:
            confidence = "Medium"
        else:
            confidence = "Low"

    return {
        "crop_name": rec.get("crop_name", crop_name),
        "disease_name": rec.get("disease_name", disease_name),
        "visible_symptoms": visible_symptoms,
        "probable_cause": probable_cause,
        "treatment": treatment,
        "prevention": prevention,
        "confidence": str(confidence),
    }


async def init_db() -> None:
    """Initialize the SQLite database and create tables if they don't exist.

    Creates the predictions table with columns for all prediction data
    including the JSON-serialized recommendation.
    """
    # Ensure the directory exists
    db_dir = Path(DB_PATH).parent
    db_dir.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                filename TEXT NOT NULL,
                disease TEXT NOT NULL,
                confidence REAL NOT NULL,
                severity TEXT NOT NULL,
                recommendation_json TEXT NOT NULL,
                region TEXT
            )
        """)
        # Create index on timestamp for efficient history queries
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_timestamp
            ON predictions (timestamp DESC)
        """)
        await db.commit()
        logger.info(f"Database initialized at {DB_PATH}")


async def save_prediction(
    filename: str,
    disease: str,
    confidence: float,
    severity: str,
    recommendation: dict,
    region: Optional[str] = None,
) -> int:
    """Save a prediction result to the database.

    Args:
        filename: Original uploaded image filename.
        disease: Detected disease name (human-readable).
        confidence: Model confidence score (0–1).
        severity: Disease severity level.
        recommendation: Structured recommendation dictionary.
        region: Optional geographic region.

    Returns:
        The database row ID of the saved record.
    """
    timestamp = datetime.utcnow().isoformat()
    recommendation_json = json.dumps(recommendation, ensure_ascii=False)

    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            """
            INSERT INTO predictions (timestamp, filename, disease, confidence, severity, recommendation_json, region)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (timestamp, filename, disease, confidence, severity, recommendation_json, region),
        )
        await db.commit()
        record_id = cursor.lastrowid
        logger.info(f"Prediction saved: id={record_id}, disease={disease}, severity={severity}")
        return record_id


async def get_history(
    page: int = 1,
    per_page: int = 20,
) -> dict:
    """Retrieve prediction history with pagination.

    Args:
        page: Page number (1-indexed).
        per_page: Number of records per page.

    Returns:
        Dictionary with total count, pagination info, and records list.
    """
    offset = (page - 1) * per_page

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        # Get total count
        async with db.execute("SELECT COUNT(*) as count FROM predictions") as cursor:
            row = await cursor.fetchone()
            total = row[0] if row else 0

        # Get paginated records
        async with db.execute(
            """
            SELECT id, timestamp, filename, disease, confidence, severity, recommendation_json, region
            FROM predictions
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
            """,
            (per_page, offset),
        ) as cursor:
            rows = await cursor.fetchall()

        records = []
        for row in rows:
            try:
                recommendation = json.loads(row[6])  # recommendation_json
            except (json.JSONDecodeError, TypeError):
                recommendation = {}

            # Normalize old-format records to current schema
            disease_str = row[3] or ""
            recommendation = _normalize_recommendation(recommendation, disease_str)

            records.append({
                "id": row[0],
                "timestamp": row[1],
                "filename": row[2],
                "disease": disease_str,
                "confidence": row[4],
                "severity": row[5],
                "recommendation": recommendation,
                "region": row[7],
            })

        return {
            "total": total,
            "page": page,
            "per_page": per_page,
            "records": records,
        }


async def get_prediction_by_id(prediction_id: int) -> Optional[dict]:
    """Retrieve a single prediction by its ID.

    Args:
        prediction_id: Database row ID.

    Returns:
        Prediction record dictionary, or None if not found.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT id, timestamp, filename, disease, confidence, severity, recommendation_json, region
            FROM predictions
            WHERE id = ?
            """,
            (prediction_id,),
        ) as cursor:
            row = await cursor.fetchone()

        if row is None:
            return None

        try:
            recommendation = json.loads(row[6])
        except (json.JSONDecodeError, TypeError):
            recommendation = {
                "description": "Recommendation data unavailable.",
                "causes": "N/A",
                "treatment": "N/A",
                "prevention": "N/A",
            }

        return {
            "id": row[0],
            "timestamp": row[1],
            "filename": row[2],
            "disease": row[3],
            "confidence": row[4],
            "severity": row[5],
            "recommendation": recommendation,
            "region": row[7],
        }
