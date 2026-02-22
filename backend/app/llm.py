"""
Vision-Language Model (VLM) Recommendation Module.

Uses LLaVA-Phi-3-Mini (INT4 quantized GGUF, ~2.3GB) to analyze crop leaf images
directly and generate structured diagnostic reports with treatment recommendations.

The VLM receives the actual image via Llava15ChatHandler (CLIP vision encoder) --
no separate CNN classification needed for the recommendation step.  The CNN still
runs first for a quick label/confidence, but the VLM provides the detailed
analysis from visual evidence.

Falls back to a hardcoded expert recommendation dictionary when VLM is unavailable.
"""

import base64
import io
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Optional

from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)


# ======================================================================
# FALLBACK RECOMMENDATIONS -- Expert-curated per-disease dictionary
# ======================================================================

FALLBACK_RECOMMENDATIONS = {
    "Apple___Apple_scab": {
        "crop_name": "Apple",
        "disease_name": "Apple Scab",
        "visible_symptoms": "Olive-green to dark brown lesions on leaves and fruit surfaces with velvety texture. Leaves may curl and drop prematurely.",
        "probable_cause": "Fungal infection by Venturia inaequalis. Favored by cool, wet spring weather (15-25 C) and prolonged leaf wetness.",
        "treatment": {
            "chemical_control": "Apply captan, mancozeb, or myclobutanil fungicides at green tip stage through petal fall. Repeat at 7-10 day intervals during wet weather.",
            "organic_control": "Spray sulfur or copper-based fungicides preventively. Apply Bacillus subtilis biofungicide. Neem oil can supplement control."
        },
        "prevention": "Plant scab-resistant varieties (Liberty, Enterprise). Remove fallen leaf litter in autumn. Prune for air circulation. Apply preventive sprays before rain.",
        "confidence": "High"
    },
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot": {
        "crop_name": "Corn (Maize)",
        "disease_name": "Gray Leaf Spot",
        "visible_symptoms": "Rectangular, grayish-tan lesions running parallel to leaf veins. Lesions 2-8 cm long with sharp, parallel edges.",
        "probable_cause": "Fungal infection by Cercospora zeae-maydis. Favored by warm temperatures (25-30 C), high humidity, and conservation tillage with corn residue.",
        "treatment": {
            "chemical_control": "Apply strobilurin or triazole-based foliar fungicides at VT/R1 stage when lesions reach the third leaf below the ear.",
            "organic_control": "No widely effective biological control exists. Focus on cultural methods: crop rotation and residue management."
        },
        "prevention": "Use resistant corn hybrids. Rotate away from corn for at least one year. Manage residue through tillage. Avoid excessive nitrogen.",
        "confidence": "High"
    },
    "Corn_(maize)___Common_rust": {
        "crop_name": "Corn (Maize)",
        "disease_name": "Common Rust",
        "visible_symptoms": "Small, circular to elongated reddish-brown pustules on both leaf surfaces. Pustules rupture to release powdery spores.",
        "probable_cause": "Fungal infection by Puccinia sorghi. Favored by moderate temperatures (16-23 C) and high humidity. Spores are wind-dispersed.",
        "treatment": {
            "chemical_control": "Apply triazole or strobilurin foliar fungicides if pustules appear on 50%+ of plants before tasseling.",
            "organic_control": "Most modern hybrids have adequate resistance. Remove severely infected plants. No effective organic fungicide available."
        },
        "prevention": "Plant resistant hybrids. Monitor fields in cool, humid conditions. Plant early to avoid peak spore dispersal.",
        "confidence": "High"
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "crop_name": "Corn (Maize)",
        "disease_name": "Northern Leaf Blight",
        "visible_symptoms": "Long, elliptical, grayish-green to tan lesions (2.5-15 cm) on leaves. Lesions may coalesce, killing large leaf areas.",
        "probable_cause": "Fungal infection by Exserohilum turcicum. Favored by moderate temperatures (18-27 C), heavy dews, and frequent rainfall.",
        "treatment": {
            "chemical_control": "Apply strobilurin, triazole, or combination fungicides at VT/R1 if disease threatens ear leaf during grain fill.",
            "organic_control": "Crop rotation and residue burial reduce inoculum. No established biological control agent."
        },
        "prevention": "Plant hybrids with NLB resistance genes (Ht1, Ht2, Ht3). Rotate with non-host crops. Reduce surface residue.",
        "confidence": "High"
    },
    "Grape___Black_rot": {
        "crop_name": "Grape",
        "disease_name": "Black Rot",
        "visible_symptoms": "Brown circular leaf lesions with dark borders. Fruit shrivels into hard, black mummified berries.",
        "probable_cause": "Fungal infection by Guignardia bidwellii. Favored by warm (21-32 C), wet conditions. Requires 6+ hours of leaf wetness.",
        "treatment": {
            "chemical_control": "Apply myclobutanil, mancozeb, or captan from bud break through veraison. Critical period: bloom through 4 weeks post-bloom.",
            "organic_control": "Remove mummified berries and infected canes. Improve airflow through canopy management. Copper sprays at dormancy."
        },
        "prevention": "Remove mummies and prune infected wood. Use canopy management for air circulation. Apply protective fungicides on schedule.",
        "confidence": "High"
    },
    "Grape___Esca_(Black_Measles)": {
        "crop_name": "Grape",
        "disease_name": "Esca (Black Measles)",
        "visible_symptoms": "Tiger-stripe patterns on leaves with interveinal chlorosis and necrosis. Dark spots on berries. Vines may show sudden wilting.",
        "probable_cause": "Complex of wood-inhabiting fungi (Phaeomoniella chlamydospora, Phaeoacremonium spp.). Enters through pruning wounds.",
        "treatment": {
            "chemical_control": "No curative chemical treatment available. Apply wound protectants after pruning.",
            "organic_control": "Apply Trichoderma harzianum to pruning wounds. Remove severely affected vines. Delay pruning to late winter."
        },
        "prevention": "Protect pruning wounds with biological sealants. Avoid large pruning cuts. Use double pruning technique. Monitor for early symptoms.",
        "confidence": "Medium"
    },
    "Potato___Early_blight": {
        "crop_name": "Potato",
        "disease_name": "Early Blight",
        "visible_symptoms": "Dark brown lesions with concentric rings (target spots) on older leaves. Progressive defoliation from bottom upward.",
        "probable_cause": "Fungal infection by Alternaria solani. Favored by warm temperatures (24-29 C) and alternating wet/dry periods. Stressed plants more susceptible.",
        "treatment": {
            "chemical_control": "Apply chlorothalonil or mancozeb preventively. Use azoxystrobin or difenoconazole under high disease pressure. Rotate fungicide classes.",
            "organic_control": "Apply Bacillus subtilis biofungicide. Neem-based products. Remove infected lower leaves. Ensure adequate plant nutrition."
        },
        "prevention": "Use certified disease-free seed potatoes. Practice 2-3 year crop rotation. Maintain balanced fertility. Use resistant varieties.",
        "confidence": "High"
    },
    "Potato___Late_blight": {
        "crop_name": "Potato",
        "disease_name": "Late Blight",
        "visible_symptoms": "Water-soaked lesions rapidly enlarging to brown-black areas. White sporulation on leaf undersides in humid conditions. Can destroy fields in days.",
        "probable_cause": "Oomycete Phytophthora infestans. Favored by cool (10-24 C), wet conditions with >90% humidity. Spreads explosively via wind.",
        "treatment": {
            "chemical_control": "Apply mancozeb/chlorothalonil (protectant) + metalaxyl or cymoxanil (systemic) on 5-7 day schedule. Start before disease appears.",
            "organic_control": "Copper-based sprays. Remove and destroy infected plants immediately. Harvest in dry conditions."
        },
        "prevention": "Use certified seed. Plant resistant varieties. Destroy cull piles. Hill tubers. Monitor weather with blight forecasting systems.",
        "confidence": "High"
    },
    "Tomato___Bacterial_spot": {
        "crop_name": "Tomato",
        "disease_name": "Bacterial Spot",
        "visible_symptoms": "Small, dark, water-soaked spots on leaves with yellow halos. Raised scabby spots on fruit. Progressive defoliation.",
        "probable_cause": "Bacterial infection by Xanthomonas spp. Favored by warm (24-30 C) wet conditions. Spreads via rain splash and contaminated seed.",
        "treatment": {
            "chemical_control": "Apply copper hydroxide combined with mancozeb. Note: copper-resistant strains are common in some areas.",
            "organic_control": "Bacillus-based biocontrol agents. Remove heavily infected plants. Avoid overhead irrigation."
        },
        "prevention": "Use pathogen-free seed. Hot water seed treatment (50 C for 25 min). Avoid working in wet fields. Rotate 2-3 years with non-solanaceous crops.",
        "confidence": "High"
    },
    "Tomato___Early_blight": {
        "crop_name": "Tomato",
        "disease_name": "Early Blight",
        "visible_symptoms": "Dark brown lesions with concentric rings (target spots) on lower leaves. Progressive defoliation upward. Stem lesions possible.",
        "probable_cause": "Fungal infection by Alternaria solani. Favored by warm (24-29 C) humid conditions. Overwinters in plant debris.",
        "treatment": {
            "chemical_control": "Apply chlorothalonil or mancozeb preventively. Use azoxystrobin or difenoconazole when active. Rotate fungicide classes.",
            "organic_control": "Trichoderma or Bacillus subtilis soil/foliar applications. Neem oil. Remove infected lower leaves. Mulch to prevent splash."
        },
        "prevention": "Practice 3-year rotation. Remove all debris after harvest. Use resistant varieties. Mulch plants. Maintain balanced nutrition.",
        "confidence": "High"
    },
    "Tomato___Late_blight": {
        "crop_name": "Tomato",
        "disease_name": "Late Blight",
        "visible_symptoms": "Large, irregularly shaped, water-soaked greenish-black lesions. White fuzzy growth on leaf undersides. Rapid plant death.",
        "probable_cause": "Oomycete Phytophthora infestans. Favored by cool (10-24 C) wet/humid conditions. Spreads rapidly via airborne sporangia.",
        "treatment": {
            "chemical_control": "Apply mancozeb/chlorothalonil + metalaxyl or cymoxanil. Spray every 5-7 days under high pressure.",
            "organic_control": "Copper-based sprays. Remove and destroy infected plants immediately. Avoid overhead irrigation."
        },
        "prevention": "Use resistant varieties (Ph-2, Ph-3 genes). Keep away from potato fields. Destroy volunteers. Monitor weather conditions.",
        "confidence": "High"
    },
    "Tomato___Leaf_Mold": {
        "crop_name": "Tomato",
        "disease_name": "Leaf Mold",
        "visible_symptoms": "Pale green/yellowish spots on upper leaf surface. Olive-green to grayish-purple fuzzy mold on leaf undersides.",
        "probable_cause": "Fungal infection by Passalora fulva. Favored by high humidity (>85%), moderate temperatures (22-30 C). Primarily greenhouse disease.",
        "treatment": {
            "chemical_control": "Apply chlorothalonil, mancozeb, or copper fungicides. Sulfur vaporization in greenhouses.",
            "organic_control": "Bacillus amyloliquefaciens biocontrol. Increase ventilation. Remove infected leaves. Reduce humidity."
        },
        "prevention": "Use resistant varieties with Cf resistance genes. Maintain humidity below 85%. Use fans for circulation. Drip irrigation only.",
        "confidence": "High"
    },
    "Tomato___Septoria_leaf_spot": {
        "crop_name": "Tomato",
        "disease_name": "Septoria Leaf Spot",
        "visible_symptoms": "Numerous small circular spots (2-5 mm) with dark brown borders and grayish-white centers. Tiny black pycnidia in spot centers.",
        "probable_cause": "Fungal infection by Septoria lycopersici. Favored by warm (20-25 C) wet conditions. Spreads via rain splash from debris.",
        "treatment": {
            "chemical_control": "Apply chlorothalonil, mancozeb, or copper fungicides early and regularly. Use azoxystrobin for rapid spread.",
            "organic_control": "Copper combined with biological agents. Remove infected lower leaves. Stake and mulch to reduce splash."
        },
        "prevention": "Crop rotation 2-3 years. Remove all debris after harvest. Use clean transplants. Mulch plants. Avoid overhead irrigation.",
        "confidence": "High"
    },
    "Tomato___Target_Spot": {
        "crop_name": "Tomato",
        "disease_name": "Target Spot",
        "visible_symptoms": "Concentric ring lesions on leaves, stems, and fruit. Lesions may coalesce causing extensive defoliation.",
        "probable_cause": "Fungal infection by Corynespora cassiicola. Favored by warm (20-30 C) humid conditions with extended leaf wetness.",
        "treatment": {
            "chemical_control": "Apply chlorothalonil, mancozeb, or azoxystrobin. Rotate fungicide classes to prevent resistance.",
            "organic_control": "Trichoderma-based products. Remove infected material. Improve spacing and air circulation."
        },
        "prevention": "Use resistant varieties. Crop rotation with non-host crops. Remove debris promptly. Ensure good air movement.",
        "confidence": "High"
    },
    "Tomato___Yellow_Leaf_Curl_Virus": {
        "crop_name": "Tomato",
        "disease_name": "Yellow Leaf Curl Virus (TYLCV)",
        "visible_symptoms": "Severe stunting. Upward curling and yellowing of leaves. Flower drop. Drastically reduced fruit set.",
        "probable_cause": "Viral infection (Begomovirus) transmitted by whiteflies (Bemisia tabaci). No cure for infected plants.",
        "treatment": {
            "chemical_control": "Apply imidacloprid or cyantraniliprole to control whitefly vectors. Remove and destroy infected plants.",
            "organic_control": "Release Encarsia formosa or Eretmocerus parasitoids. Use reflective mulches and yellow sticky traps for whiteflies."
        },
        "prevention": "Plant TYLCV-resistant varieties (Ty-1, Ty-2, Ty-3 genes). Install insect screens in greenhouses. Control weed hosts. Start with virus-free transplants.",
        "confidence": "High"
    },
    "Healthy": {
        "crop_name": "Unknown",
        "disease_name": "Healthy",
        "visible_symptoms": "No visible disease symptoms. Leaves show normal color, shape, and vigor.",
        "probable_cause": "No disease detected. Plant appears healthy.",
        "treatment": {
            "chemical_control": "No treatment needed.",
            "organic_control": "Continue regular care: adequate watering, balanced fertilization, and routine monitoring."
        },
        "prevention": "Maintain good agricultural practices: proper spacing, balanced nutrition, regular scouting, crop rotation, and tool sanitation.",
        "confidence": "High"
    },
}


# ======================================================================
# VLM SYSTEM PROMPT -- Expert agricultural pathologist
# ======================================================================

# Raw text prompt template — no chat handler overhead
RAW_PROMPT_TEMPLATE = (
    "Disease: {disease}. Severity: {severity}.\n"
    "Chemical treatment: "
)


# ======================================================================
# VLM ENGINE -- LLaVA-Phi-3-Mini GGUF via Llava15ChatHandler
# ======================================================================

class RecommendationEngine:
    """Vision-Language Model based recommendation generator.

    Uses LLaVA-Phi-3-Mini (INT4 GGUF) with CLIP vision projector to analyze
    crop images directly and produce structured diagnostic JSON.
    Falls back to a curated dictionary when the VLM is unavailable.
    """

    def __init__(self):
        self.backend: Optional[str] = None
        self.model = None
        self.chat_handler = None
        self.is_loaded: bool = False
        self._prompt_log: list = []

    def load(self) -> bool:
        """Load the VLM model.

        Returns:
            True if loaded, False if falling back to dictionary.
        """
        backend = settings.LLM_BACKEND.lower()
        logger.info(f"Attempting to load VLM with backend: {backend}")

        if backend == "gguf":
            return self._load_gguf()
        else:
            logger.warning(f"Unknown VLM backend '{backend}'. Using fallback.")
            return False

    def _load_gguf(self) -> bool:
        """Load LLaVA-Phi-3-Mini GGUF model with CLIP vision projection."""
        model_path = Path(settings.LLM_GGUF_PATH)
        mmproj_path = Path(settings.LLM_MMPROJ_PATH)

        if not model_path.exists():
            logger.warning(
                f"VLM GGUF not found at {model_path}. "
                "Using fallback recommendations. "
                "Run download script to get llava-phi-3-mini-int4.gguf."
            )
            return False

        if not mmproj_path.exists():
            logger.warning(
                f"CLIP projection not found at {mmproj_path}. "
                "Using fallback recommendations. "
                "Run download script to get llava-phi-3-mini-mmproj-f16.gguf."
            )
            return False

        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Llava15ChatHandler

            logger.info(f"Loading LLaVA-Phi-3-Mini VLM from {model_path}...")
            logger.info(f"Loading CLIP projector from {mmproj_path}...")

            self.chat_handler = Llava15ChatHandler(
                clip_model_path=str(mmproj_path),
                verbose=False,
            )

            self.model = Llama(
                model_path=str(model_path),
                chat_handler=self.chat_handler,
                n_ctx=settings.LLM_CONTEXT_LENGTH,
                n_threads=os.cpu_count() or 4,
                n_gpu_layers=0,  # CPU-only by default
                n_batch=256,     # larger batch = faster prompt eval
                use_mmap=True,
                verbose=False,
            )
            self.backend = "gguf"
            self.is_loaded = True
            logger.info("LLaVA-Phi-3-Mini (INT4) loaded successfully.")

            # Warmup: run a tiny raw inference to prime KV cache
            try:
                logger.info("Warming up LLM (first-token latency fix)...")
                self.model(
                    "Hello",
                    max_tokens=1,
                    echo=False,
                )
                logger.info("LLM warmup complete.")
            except Exception as we:
                logger.warning(f"LLM warmup failed (non-fatal): {we}")

            return True

        except ImportError as e:
            logger.warning(f"llama-cpp-python import error: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load GGUF VLM: {e}", exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_recommendation(
        self,
        disease: str,
        confidence: float,
        severity: str,
        region: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
    ) -> dict:
        """Generate a structured recommendation.

        If VLM is loaded and image_bytes are provided, the VLM analyzes
        the image directly.  Otherwise falls back to curated dictionary.

        Args:
            disease: CNN-predicted disease display name.
            confidence: CNN confidence (0-1).
            severity: Severity level string.
            region: Optional geographic region.
            image_bytes: Raw image bytes for VLM analysis.

        Returns:
            Dict matching the diagnostic JSON schema.
        """
        if self.is_loaded and self.model is not None and settings.VLM_ENABLED:
            try:
                # Use text-only LLM mode (fast — skips CLIP vision encoder).
                # LLM generates only treatment text, merged into fallback dict.
                result = self._generate_text_only(
                    disease, confidence, severity, region,
                )
                if result:
                    return result
            except Exception as e:
                logger.error(f"LLM generation failed: {e}", exc_info=True)

        logger.info(f"Using fallback recommendation for: {disease}")
        return self._get_fallback(disease, severity)

    @staticmethod
    def _merge_vlm_with_fallback(vlm: dict, fallback: dict) -> dict:
        """Fill empty/short VLM fields with expert fallback data."""
        for key in ("crop_name", "disease_name", "visible_symptoms",
                    "probable_cause", "prevention", "confidence"):
            vlm_val = vlm.get(key, "")
            if not vlm_val or len(str(vlm_val)) < 5:
                vlm[key] = fallback.get(key, vlm_val)

        # Merge treatment sub-fields
        vlm_treat = vlm.get("treatment", {})
        fb_treat = fallback.get("treatment", {})
        if isinstance(vlm_treat, dict) and isinstance(fb_treat, dict):
            for sub in ("chemical_control", "organic_control"):
                if not vlm_treat.get(sub) or len(vlm_treat.get(sub, "")) < 10:
                    vlm_treat[sub] = fb_treat.get(sub, vlm_treat.get(sub, ""))
            vlm["treatment"] = vlm_treat

        return vlm

    # ------------------------------------------------------------------
    # Text-only LLM inference (FAST — no CLIP vision encoding)
    # ------------------------------------------------------------------

    def _generate_text_only(
        self,
        disease: str,
        confidence: float,
        severity: str,
        region: Optional[str] = None,
    ) -> Optional[dict]:
        """Generate treatment using raw model() completion (no chat handler).

        Uses raw text completion which is MUCH faster than
        create_chat_completion() because it bypasses the Llava15ChatHandler
        entirely. Typically 10-20s on CPU instead of 45-60s.
        """
        prompt = RAW_PROMPT_TEMPLATE.format(disease=disease, severity=severity)

        if settings.LOG_PROMPTS:
            self._prompt_log.append({
                "mode": "raw-completion",
                "disease_hint": disease,
                "severity": severity,
                "prompt": prompt,
            })

        logger.info(
            f"Generating treatment for '{disease}' via raw LLM completion "
            f"(timeout={settings.VLM_TIMEOUT}s, max_tokens={settings.LLM_MAX_TOKENS})..."
        )

        def _call_llm():
            return self.model(
                prompt,
                max_tokens=settings.LLM_MAX_TOKENS,
                temperature=settings.LLM_TEMPERATURE,
                top_p=1.0,
                echo=False,
                stop=["\n\n", "Prevention:", "Note:"],
            )

        pool = ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(_call_llm)
            response = future.result(timeout=settings.VLM_TIMEOUT)
        except FuturesTimeoutError:
            logger.warning(
                f"LLM inference timed out after {settings.VLM_TIMEOUT}s. "
                "Falling back to curated recommendation."
            )
            pool.shutdown(wait=False, cancel_futures=True)
            return None
        except Exception as e:
            logger.error(f"LLM inference error: {e}", exc_info=True)
            pool.shutdown(wait=False, cancel_futures=True)
            return None
        else:
            pool.shutdown(wait=False)

        raw_output = response["choices"][0]["text"].strip()
        logger.info(f"LLM output length: {len(raw_output)} chars")
        logger.info(f"LLM raw output: {raw_output[:300]}")

        return self._parse_treatment_text(raw_output, disease, severity)

    def _parse_treatment_text(
        self, text: str, disease: str, severity: str
    ) -> Optional[dict]:
        """Parse LLM treatment text into a full recommendation dict.

        Tries to detect CHEMICAL:/ORGANIC: prefixes. If not found,
        splits at 'organic' keyword. Falls back to using full text as chemical.
        Merges into the expert fallback dictionary.
        """
        if not text or len(text) < 10:
            return None

        chemical = ""
        organic = ""

        # Try labeled format: "CHEMICAL: ... ORGANIC: ..."
        for line in text.split("\n"):
            stripped = line.strip()
            upper = stripped.upper()
            if upper.startswith("CHEMICAL"):
                chemical = stripped.split(":", 1)[-1].strip()
            elif upper.startswith("ORGANIC"):
                organic = stripped.split(":", 1)[-1].strip()

        # If no labels found, use heuristic split
        if not chemical and not organic:
            lower = text.lower()
            idx = lower.find("organic")
            if idx > 0:
                chemical = text[:idx].strip().rstrip(",.-;:")
                organic = text[idx:].strip()
                # Remove the word "organic" prefix if it starts a sentence
                if organic.lower().startswith("organic"):
                    organic = organic[7:].strip().lstrip(":").strip()
            else:
                # No split possible — use full text as chemical treatment
                chemical = text.strip()

        # Build result using fallback as base
        result = self._get_fallback(disease, severity)
        if chemical and len(chemical) > 5:
            result["treatment"]["chemical_control"] = chemical
        if organic and len(organic) > 5:
            result["treatment"]["organic_control"] = organic

        return result

    # ------------------------------------------------------------------
    # VLM inference (vision mode — slow on CPU, fast with GPU)
    # ------------------------------------------------------------------

    @staticmethod
    def _resize_image(image_bytes: bytes, max_size: int) -> bytes:
        """Resize an image to max_size x max_size to reduce CLIP processing load."""
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img = img.convert("RGB")
            img.thumbnail((max_size, max_size), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            resized = buf.getvalue()
            logger.info(
                f"Image resized: {len(image_bytes)//1024}KB -> "
                f"{len(resized)//1024}KB ({img.size[0]}x{img.size[1]})"
            )
            return resized
        except Exception as e:
            logger.warning(f"Image resize failed ({e}), using original")
            return image_bytes

    def _generate_with_vlm(
        self,
        image_bytes: bytes,
        disease: str,
        confidence: float,
        severity: str,
        region: Optional[str] = None,
    ) -> Optional[dict]:
        """Send the image to LLaVA-Phi-3-Mini and parse the diagnostic JSON."""
        # Resize image to reduce CLIP processing time on CPU
        resized = self._resize_image(image_bytes, settings.VLM_IMAGE_MAX_SIZE)
        img_b64 = base64.b64encode(resized).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{img_b64}"

        user_text = "Analyze this leaf image. Identify the disease and recommend treatment medicines."
        if disease and disease != "Healthy":
            user_text += f" CNN detected '{disease}' ({confidence:.0%}, {severity})."
        if region:
            user_text += f" Region: {region}."

        messages = [
            {"role": "system", "content": VLM_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": user_text},
                ],
            },
        ]

        if settings.LOG_PROMPTS:
            self._prompt_log.append({
                "disease_hint": disease,
                "severity": severity,
                "region": region,
                "prompt": user_text,
            })

        logger.info(
            f"Sending image to LLaVA-Phi-3-Mini for analysis "
            f"(timeout={settings.VLM_TIMEOUT}s)..."
        )

        # Run VLM inference in a thread with timeout so we can fall back
        # to the curated dictionary if it takes too long on CPU.
        def _call_vlm():
            return self.model.create_chat_completion(
                messages=messages,
                max_tokens=settings.LLM_MAX_TOKENS,
                temperature=settings.LLM_TEMPERATURE,
                top_p=0.9,
            )

        pool = ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(_call_vlm)
            response = future.result(timeout=settings.VLM_TIMEOUT)
        except FuturesTimeoutError:
            logger.warning(
                f"VLM inference timed out after {settings.VLM_TIMEOUT}s. "
                "Falling back to curated recommendation."
            )
            # Do NOT wait for the worker — let it finish in the background
            pool.shutdown(wait=False, cancel_futures=True)
            return None
        except Exception as e:
            logger.error(f"VLM inference error: {e}", exc_info=True)
            pool.shutdown(wait=False, cancel_futures=True)
            return None
        else:
            pool.shutdown(wait=False)

        raw_output = response["choices"][0]["message"]["content"].strip()
        logger.info(f"VLM output length: {len(raw_output)} chars")
        logger.debug(f"VLM output: {raw_output[:500]}")

        return self._parse_vlm_json(raw_output)

    # ------------------------------------------------------------------
    # JSON parsing
    # ------------------------------------------------------------------

    def _parse_vlm_json(self, raw_text: str) -> Optional[dict]:
        """Parse structured JSON from VLM output."""
        if not raw_text:
            return None

        # Strip markdown code fences
        cleaned = raw_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        # Direct parse
        try:
            result = json.loads(cleaned)
            if self._validate_vlm_response(result):
                return self._normalize_vlm_response(result)
        except json.JSONDecodeError:
            pass

        # Extract first JSON object
        json_pat = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        for match in re.findall(json_pat, raw_text, re.DOTALL):
            try:
                result = json.loads(match)
                if self._validate_vlm_response(result):
                    return self._normalize_vlm_response(result)
            except json.JSONDecodeError:
                continue

        logger.warning(f"Failed to parse VLM JSON: {raw_text[:300]}...")
        return None

    @staticmethod
    def _validate_vlm_response(data: dict) -> bool:
        if not isinstance(data, dict):
            return False
        required = {
            "crop_name", "disease_name", "visible_symptoms",
            "probable_cause", "treatment", "prevention", "confidence",
        }
        if not required.issubset(data.keys()):
            return False
        if isinstance(data["treatment"], dict):
            return ("chemical_control" in data["treatment"]
                    or "organic_control" in data["treatment"])
        return isinstance(data["treatment"], str)

    @staticmethod
    def _normalize_vlm_response(data: dict) -> dict:
        if isinstance(data["treatment"], str):
            data["treatment"] = {
                "chemical_control": data["treatment"],
                "organic_control": (
                    "Consult a local agricultural extension officer "
                    "for organic alternatives."
                ),
            }
        data["treatment"].setdefault("chemical_control", "")
        data["treatment"].setdefault("organic_control", "")
        return data

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def _get_fallback(self, disease: str, severity: str) -> dict:
        """Return a curated fallback recommendation."""
        rec = None
        for label, data in FALLBACK_RECOMMENDATIONS.items():
            if label == disease:
                rec = _deep_copy(data)
                break

        if rec is None:
            from app.config import DISEASE_DISPLAY_NAMES
            for label, display_name in DISEASE_DISPLAY_NAMES.items():
                if display_name == disease:
                    rec = _deep_copy(FALLBACK_RECOMMENDATIONS.get(label, {}))
                    break

        if rec is None:
            rec = {
                "crop_name": "Unknown",
                "disease_name": disease,
                "visible_symptoms": (
                    "Symptoms detected by CNN classifier. "
                    "Consult a local agricultural extension officer."
                ),
                "probable_cause": (
                    "Conditions vary by pathogen and environment. "
                    "Laboratory confirmation recommended."
                ),
                "treatment": {
                    "chemical_control": (
                        "Apply appropriate fungicide or pesticide as "
                        "recommended by local agricultural authorities."
                    ),
                    "organic_control": (
                        "Remove affected plant parts. Improve growing "
                        "conditions. Use neem-based products."
                    ),
                },
                "prevention": (
                    "Practice crop rotation, maintain proper spacing, "
                    "use disease-resistant varieties, and scout regularly."
                ),
                "confidence": "Low",
            }

        severity_notes = {
            "mild": " For mild cases, cultural controls alone may suffice.",
            "moderate": " Moderate severity -- combine cultural and chemical methods.",
            "severe": (
                " URGENT: Severe infection requires immediate chemical "
                "treatment and removal of heavily infected material."
            ),
        }
        if severity in severity_notes and isinstance(rec.get("treatment"), dict):
            rec["treatment"]["chemical_control"] += severity_notes[severity]

        return rec

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        return {
            "loaded": self.is_loaded,
            "backend": self.backend or "fallback",
            "model_name": (
                "LLaVA-Phi-3-Mini-INT4-GGUF" if self.backend == "gguf"
                else "fallback_dictionary"
            ),
            "vlm_enabled": settings.VLM_ENABLED,
            "mode": "text-only-llm" if settings.VLM_ENABLED else "fallback",
            "prompts_logged": len(self._prompt_log),
        }

    def get_prompt_log(self) -> list:
        return self._prompt_log.copy()


def _deep_copy(d: dict) -> dict:
    """Simple deep copy for dicts containing only str/dict values."""
    out = {}
    for k, v in d.items():
        out[k] = _deep_copy(v) if isinstance(v, dict) else v
    return out


# -- Module-level singleton --
recommendation_engine = RecommendationEngine()
