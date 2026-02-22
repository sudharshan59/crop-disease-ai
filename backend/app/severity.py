"""
Severity estimation module.

Combines CNN confidence scores with image-level analysis (diseased area ratio)
to determine disease severity: healthy, mild, moderate, severe.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ─── Severity Thresholds ─────────────────────────────────────────────────────
# These thresholds combine confidence and visual analysis for severity grading.
CONFIDENCE_SEVERE = 0.90   # High confidence → likely severe/clear symptoms
CONFIDENCE_MODERATE = 0.70  # Moderate confidence → moderate symptoms
# Below 0.70 confidence → mild symptoms or uncertain

AREA_SEVERE = 0.50      # >50% of leaf area affected → severe
AREA_MODERATE = 0.25    # >25% → moderate
# Below 25% → mild


def estimate_severity(
    confidence: float,
    is_healthy: bool,
    diseased_area_ratio: Optional[float] = None,
) -> str:
    """Estimate disease severity based on confidence and visual analysis.

    The severity estimation combines two signals:
    1. CNN confidence score — higher confidence often correlates with
       more pronounced/clear symptoms.
    2. Diseased area ratio — percentage of the leaf showing discoloration,
       computed via OpenCV segmentation in preprocessing.

    Severity Levels:
    - "healthy": Plant classified as healthy.
    - "mild": Early stage, minor symptoms.
    - "moderate": Noticeable symptoms requiring attention.
    - "severe": Extensive disease requiring immediate treatment.

    Args:
        confidence: Model confidence score (0.0 to 1.0).
        is_healthy: Whether the model predicted "Healthy".
        diseased_area_ratio: Optional ratio of diseased leaf area (0.0 to 1.0).

    Returns:
        Severity level string: "healthy", "mild", "moderate", or "severe".
    """
    # Healthy plants get a clean bill of health
    if is_healthy:
        return "healthy"

    # If we have both signals, use a combined scoring approach
    if diseased_area_ratio is not None:
        severity_score = _compute_combined_score(confidence, diseased_area_ratio)
        return _score_to_severity(severity_score)

    # Confidence-only estimation (no segmentation data)
    return _confidence_to_severity(confidence)


def _confidence_to_severity(confidence: float) -> str:
    """Map confidence score alone to severity level.

    Args:
        confidence: CNN confidence score (0.0 to 1.0).

    Returns:
        Severity string.
    """
    if confidence >= CONFIDENCE_SEVERE:
        return "severe"
    elif confidence >= CONFIDENCE_MODERATE:
        return "moderate"
    else:
        return "mild"


def _compute_combined_score(confidence: float, area_ratio: float) -> float:
    """Compute a blended severity score from confidence and area ratio.

    Weights:
    - Confidence: 40% (model certainty)
    - Area ratio: 60% (visual severity)

    Args:
        confidence: CNN confidence (0–1).
        area_ratio: Diseased area ratio (0–1).

    Returns:
        Combined score between 0.0 and 1.0.
    """
    WEIGHT_CONFIDENCE = 0.4
    WEIGHT_AREA = 0.6

    combined = (confidence * WEIGHT_CONFIDENCE) + (area_ratio * WEIGHT_AREA)
    return max(0.0, min(1.0, combined))


def _score_to_severity(score: float) -> str:
    """Convert a combined severity score to a severity level.

    Args:
        score: Combined severity score (0–1).

    Returns:
        Severity string.
    """
    if score >= 0.65:
        return "severe"
    elif score >= 0.40:
        return "moderate"
    else:
        return "mild"


def get_severity_description(severity: str) -> str:
    """Get a human-readable description for a severity level.

    Args:
        severity: Severity level string.

    Returns:
        Description text.
    """
    descriptions = {
        "healthy": "The plant appears healthy with no visible signs of disease.",
        "mild": "Early-stage infection detected. Minor symptoms visible. Early intervention recommended.",
        "moderate": "Moderate infection detected. Clear symptoms present. Prompt treatment advised.",
        "severe": "Severe infection detected. Extensive symptoms visible. Immediate action required.",
    }
    return descriptions.get(severity, "Unknown severity level.")


def get_severity_color(severity: str) -> str:
    """Get the associated color for a severity level (for UI rendering).

    Args:
        severity: Severity level string.

    Returns:
        CSS-compatible color string.
    """
    colors = {
        "healthy": "#22c55e",   # green-500
        "mild": "#eab308",      # yellow-500
        "moderate": "#f97316",  # orange-500
        "severe": "#ef4444",    # red-500
    }
    return colors.get(severity, "#6b7280")  # gray-500 fallback
