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
        treatment_params: Optional[dict] = None,
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
        # If model should be used, attempt load once more before final fallback
        if settings.VLM_ENABLED and not self.is_loaded:
            logger.info("VLM enabled but not loaded - attempting to load model...")
            self.load()

        if self.is_loaded and self.model is not None and settings.VLM_ENABLED:
            try:
                # Use text-only LLM mode (fast — skips CLIP vision encoder).
                # LLM generates only treatment text, merged into fallback dict.
                result = self._generate_text_only(
                    disease, confidence, severity, region, treatment_params,
                )
                if result:
                    return result
            except Exception as e:
                logger.error(f"LLM generation failed: {e}", exc_info=True)

        logger.info(f"Using fallback recommendation for: {disease}")
        fb = self._get_fallback(disease, severity)
        # Attach treatment params so structured builders can use them
        if treatment_params and isinstance(fb, dict):
            try:
                fb["_treatment_params"] = treatment_params
            except Exception:
                pass
        return fb

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
        treatment_params: Optional[dict] = None,
    ) -> Optional[dict]:
        """Generate treatment using raw model() completion (no chat handler).

        Uses raw text completion which is MUCH faster than
        create_chat_completion() because it bypasses the Llava15ChatHandler
        entirely. Typically 10-20s on CPU instead of 45-60s.
        """
        prompt = RAW_PROMPT_TEMPLATE.format(disease=disease, severity=severity)
        # If the caller provided explicit treatment/mixing params, include them
        if treatment_params and isinstance(treatment_params, dict):
            try:
                parts = []
                if treatment_params.get("chemical_amount_g") is not None:
                    parts.append(f"chemical_amount_g={treatment_params.get('chemical_amount_g')}")
                if treatment_params.get("water_amount_l") is not None:
                    parts.append(f"water_amount_l={treatment_params.get('water_amount_l')}")
                if treatment_params.get("organic_amount_g") is not None:
                    parts.append(f"organic_amount_g={treatment_params.get('organic_amount_g')}")
                if treatment_params.get("notes"):
                    parts.append(f"notes={treatment_params.get('notes')}")
                if parts:
                    prompt += "\n\nTreatment parameters: " + ", ".join(parts) + ". Please tailor mixing and application instructions accordingly."
            except Exception:
                pass

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

        # Ensure model-provided short fields are expanded to ~200 words
        try:
            _ensure_min_words_in_rec(result, min_words=200)
        except Exception:
            pass

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
        treatment_params: Optional[dict] = None,
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
        if treatment_params and isinstance(treatment_params, dict):
            try:
                tp = ", ".join(f"{k}={v}" for k, v in treatment_params.items())
                user_text += f" Treatment parameters: {tp}. Please tailor mixing and application instructions accordingly."
            except Exception:
                pass

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
                norm = self._normalize_vlm_response(result)
                # Merge with fallback for missing/short fields
                fb = self._get_fallback(norm.get("disease_name", ""), "")
                merged = self._merge_vlm_with_fallback(norm, fb)
                try:
                    _ensure_min_words_in_rec(merged, min_words=200)
                except Exception:
                    pass
                return merged
        except json.JSONDecodeError:
            pass

        # Extract first JSON object
        json_pat = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        for match in re.findall(json_pat, raw_text, re.DOTALL):
            try:
                result = json.loads(match)
                if self._validate_vlm_response(result):
                            norm = self._normalize_vlm_response(result)
                            fb = self._get_fallback(norm.get("disease_name", ""), "")
                            merged = self._merge_vlm_with_fallback(norm, fb)
                            try:
                                _ensure_min_words_in_rec(merged, min_words=200)
                            except Exception:
                                pass
                            return merged
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

        # Generate longer, ~200-word disease description for richer client display
        try:
            rec["disease_description"] = _generate_200_word_disease_description(disease, rec)
        except Exception:
            rec["disease_description"] = _generate_200_word_disease_description(disease, rec)
        # Capture original short texts before expansion for concise bullets
        try:
            short_texts = {
                "visible_symptoms": rec.get("visible_symptoms", ""),
                "probable_cause": rec.get("probable_cause", ""),
                "prevention": rec.get("prevention", ""),
                "chemical_control": (rec.get("treatment") or {}).get("chemical_control", ""),
                "organic_control": (rec.get("treatment") or {}).get("organic_control", ""),
            }
            # keep original short texts accessible for structured parsing
            rec["_short_texts"] = short_texts
        except Exception:
            short_texts = {}

        # Ensure each user-visible description field is expanded to ~200 words
        try:
            def _expand_field_to_200(text: str, fallback: str) -> str:
                src = text if text and isinstance(text, str) and text.strip() else fallback
                return _expand_to_n_words(src, disease)

            # Expand main fields
            rec["visible_symptoms"] = _expand_field_to_200(
                rec.get("visible_symptoms", ""),
                "Symptoms include leaf spots, discoloration, and necrosis.",
            )
            rec["probable_cause"] = _expand_field_to_200(
                rec.get("probable_cause", ""),
                "It is commonly caused by fungal, bacterial, or viral pathogens under conducive environmental conditions.",
            )
            rec["prevention"] = _expand_field_to_200(
                rec.get("prevention", ""),
                "Preventive measures include cultural practices, resistant varieties, and timely chemical or biological controls.",
            )

            # Expand treatment subfields when present
            if isinstance(rec.get("treatment"), dict):
                rec["treatment"]["chemical_control"] = _expand_field_to_200(
                    rec["treatment"].get("chemical_control", ""),
                    "Apply appropriate fungicide or pesticide as recommended by local agricultural authorities.",
                )
                rec["treatment"]["organic_control"] = _expand_field_to_200(
                    rec["treatment"].get("organic_control", ""),
                    "Remove affected plant parts, improve air circulation, and use neem-based products or biological controls.",
                )
        except Exception:
            # If expansion fails, keep original rec values
            pass
        # Build structured sections for UI (title, short summary, long detail, bullets)
        try:
            def _short_summary(long_text: str, n_words: int = 30) -> str:
                if not long_text or not isinstance(long_text, str):
                    return ""
                words = long_text.split()
                out = " ".join(words[:n_words])
                return out + ("..." if len(words) > n_words else "")

            def _bullets_from_text(text: str) -> list:
                if not text or not isinstance(text, str):
                    return []
                # Prefer splitting the original short text for concise bullets
                parts = [s.strip() for s in re.split(r"[\n\.]+", text) if s.strip()]
                if len(parts) <= 1:
                    parts = [p.strip() for p in re.split(r",|;", text) if p.strip()]
                # Clean bullets: remove overly short or duplicate items
                cleaned = []
                seen = set()
                for p in parts:
                    p = p.strip()
                    if len(p) < 4:
                        continue
                    key = p.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    cleaned.append(p)
                    if len(cleaned) >= 6:
                        break
                return cleaned

            sections = []
            vis_long = rec.get("visible_symptoms", "")
            cause_long = rec.get("probable_cause", "")
            prev_long = rec.get("prevention", "")
            chem_long = rec.get("treatment", {}).get("chemical_control", "")
            org_long = rec.get("treatment", {}).get("organic_control", "")
            # concise bullets sourced from original short texts
            vis_short = short_texts.get("visible_symptoms", "")
            cause_short = short_texts.get("probable_cause", "")
            prev_short = short_texts.get("prevention", "")
            chem_short = short_texts.get("chemical_control", "")
            org_short = short_texts.get("organic_control", "")

            sections.append({
                "id": "symptoms",
                "title": "Symptoms",
                "summary": _short_summary(vis_long),
                "detail": _synthesize_section_detail(vis_short, vis_long, disease),
                "bullets": _bullets_from_text(vis_short),
            })
            sections.append({
                "id": "cause",
                "title": "Cause",
                "summary": _short_summary(cause_long),
                "detail": _synthesize_section_detail(cause_short, cause_long, disease),
                "bullets": _bullets_from_text(cause_short),
            })
            sections.append({
                "id": "chemical",
                "title": "Chemical",
                "summary": _short_summary(chem_long),
                "detail": _synthesize_section_detail(chem_short, chem_long, disease),
                "bullets": _bullets_from_text(chem_short),
            })
            sections.append({
                "id": "organic",
                "title": "Organic",
                "summary": _short_summary(org_long),
                "detail": _synthesize_section_detail(org_short, org_long, disease),
                "bullets": _bullets_from_text(org_short),
            })
            sections.append({
                "id": "prevention",
                "title": "Prevention",
                "summary": _short_summary(prev_long),
                "detail": _synthesize_section_detail(prev_short, prev_long, disease),
                "bullets": _bullets_from_text(prev_short),
            })

            rec["sections"] = sections
        except Exception:
            rec.setdefault("sections", [])

        # Build a 4-week actionable plan (weekly steps)
        try:
            def _make_weekly_plan(disease_name: str, severity: str) -> list:
                # concise short actions per week (few items each)
                base_actions = [
                    [
                        "Inspect fields and remove heavily infected leaves.",
                        "Improve spacing and ventilation.",
                        "Sanitize tools and remove crop debris.",
                    ],
                    [
                        "Apply targeted chemical/biological control as recommended.",
                        "Follow label timing and safety precautions.",
                    ],
                    [
                        "Continue monitoring; reapply treatments if necessary.",
                        "Provide balanced nutrition and reduce plant stress.",
                    ],
                    [
                        "Evaluate crop response and yield impact.",
                        "Plan prevention: rotation, resistant varieties, residue management.",
                    ],
                ]
                # Modify actions by severity
                if severity == "severe":
                    base_actions[0] = [
                        "Immediately remove and destroy heavily infected plants.",
                        "Isolate affected area and limit movement.",
                        "Disinfect tools and equipment.",
                    ]
                    base_actions[1] = [
                        "Apply systemic fungicide/insecticide per label.",
                        "Protect neighboring plants with preventive spraying.",
                    ]
                    base_actions[2] = [
                        "Continue sanitation and monitoring.",
                        "Reapply treatments if necessary.",
                    ]
                    base_actions[3] = [
                        "Consult extension services for large-scale response.",
                        "Plan replanting and long-term management.",
                    ]
                elif severity == "moderate":
                    base_actions[0] = [
                        "Remove affected leaves and debris.",
                        "Improve spacing and ventilation.",
                    ]
                    base_actions[1] = [
                        "Apply foliar sprays or biocontrols per label.",
                    ]
                    base_actions[2] = [
                        "Monitor progress and reduce plant stress.",
                    ]
                    base_actions[3] = [
                        "Implement prevention: rotation, resistant varieties, residue management.",
                    ]

                plan = []
                for i, acts in enumerate(base_actions, start=1):
                    plan.append({"week": f"Week {i}", "actions": acts})
                return plan

            rec["weekly_plan"] = _make_weekly_plan(rec.get("disease_name", ""), rec.get("confidence", "Medium") if isinstance(rec.get("confidence"), str) else severity)
            # Also add weekly plan as a final section for UI
            wp_bullets = [p.get("actions") for p in rec.get("weekly_plan", [])]
            sections.append({
                "id": "weekly_plan",
                "title": "4-Week Plan",
                "summary": "Stepwise weekly actions to manage the disease",
                "detail": "\n\n".join(wp_bullets),
                "bullets": wp_bullets,
            })
            rec["sections"] = sections
        except Exception:
            rec.setdefault("weekly_plan", [])

        # Build more structured treatment breakdowns (chemical / mixing / application)
        try:
            def _build_structured_treatment(rec: dict) -> list:
                out = []
                # prefer short/original texts to avoid expanded repetitive text
                short_texts = rec.get("_short_texts", {}) or {}
                chem_short = short_texts.get("chemical_control", "") or ""
                org_short = short_texts.get("organic_control", "") or ""
                chem = chem_short if chem_short else (rec.get("treatment") or {}).get("chemical_control", "") or ""
                org = org_short if org_short else (rec.get("treatment") or {}).get("organic_control", "") or ""

                # Chemical details block
                if chem:
                    # Short summary
                    chem_summary = chem if len(chem.split()) < 30 else " ".join(chem.split()[:28]) + "..."
                    # Mixing instructions: sentences that mention quantities or mixing verbs
                    mix_parts = [s.strip() for s in re.split(r"(?<=[.!?])\\s+", chem) if s.strip()]
                    mixing = [p for p in mix_parts if re.search(r"\b(ml|l|liters|litres|g|kg|%|ratio|mix|measure|dilut|per|apply)\b", p, flags=re.I)]
                    if not mixing:
                        # fallback: take first 2 sentences as mixing/instructions candidates
                        mixing = mix_parts[:2]

                    # Application steps: sentences with action verbs
                    app_steps = [p for p in mix_parts if re.search(r"\b(apply|spray|mix|measure|ensure|reapply|use|avoid|dilut)\b", p, flags=re.I)]
                    if not app_steps:
                        app_steps = mix_parts

                    out.append({
                        "id": "chemical_details",
                        "title": "Chemical Details",
                        "summary": chem_summary,
                        "detail": chem,
                        "bullets": [],
                    })

                    # If explicit treatment params provided, prefer to show them as mixing instructions
                    tp = rec.get("_treatment_params") or rec.get("_treatment_params", None)
                    if tp and isinstance(tp, dict):
                        mix_list = []
                        try:
                            cm = tp.get("chemical_amount_g")
                            wl = tp.get("water_amount_l")
                            og = tp.get("organic_amount_g")
                            if cm is not None and wl is not None:
                                conc = None
                                try:
                                    conc = (float(cm) / (float(wl) * 1000.0)) * 100.0
                                except Exception:
                                    conc = None
                                mix_list.append(f"Measure {cm} g of chemical and mix into {wl} L of water.")
                                if conc is not None:
                                    mix_list.append(f"Approximate concentration: {round(conc,3)}%.")
                            if og is not None:
                                mix_list.append(f"Organic additive: {og} g (if using).")
                        except Exception:
                            mix_list = mixing
                        mixing = mix_list if mix_list else mixing

                    if mixing:
                        out.append({
                            "id": "mixing_instructions",
                            "title": "Mixing Instructions",
                            "summary": "How to prepare the spray mix",
                            "detail": "\n\n".join(mixing),
                            "bullets": mixing,
                        })

                    if app_steps:
                        out.append({
                            "id": "application_steps",
                            "title": "Application Steps",
                            "summary": "Stepwise application guidance",
                            "detail": "\n\n".join(app_steps),
                            "bullets": app_steps,
                        })

                # Organic/home remedies block
                if org:
                    org_parts = [s.strip() for s in re.split(r"(?<=[.!?])\\s+", org) if s.strip()]
                    out.append({
                        "id": "organic_home_remedy",
                        "title": "Organic / Home Remedy",
                        "summary": org_parts[0] if org_parts else "Organic measures",
                        "detail": "\n\n".join(org_parts),
                        "bullets": org_parts,
                    })

                return out

            structured = _build_structured_treatment(rec)
            # Append structured sections after existing sections (avoid duplicates)
            for s in structured:
                if not any(existing.get("id") == s.get("id") for existing in rec.get("sections", [])):
                    rec.setdefault("sections", []).append(s)
        except Exception:
            pass

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


def _generate_200_word_disease_description(disease: str, rec: dict) -> str:
    """Generate a ~200 word description for the disease for API clients.

    Builds a deterministic long-form description from available fields
    (visible_symptoms, probable_cause, prevention) and expands to
    approximately 200 words using safe repetition to maintain readability.
    """
    import re

    # Gather concise seeds from the recommendation to build meaningful sentences
    seeds = []
    if disease:
        seeds.append(f"{disease} is a leaf disease that affects crop health and can reduce yield.")

    if rec:
        vis = rec.get("visible_symptoms", "").strip()
        cause = rec.get("probable_cause", "").strip()
        chem = (rec.get("treatment") or {}).get("chemical_control", "").strip()
        org = (rec.get("treatment") or {}).get("organic_control", "").strip()
        prev = rec.get("prevention", "").strip()

        if vis:
            seeds.append(f"Common signs include: {vis}.")
        if cause:
            seeds.append(f"The primary causes are often: {cause}.")
        if chem:
            seeds.append(f"Recommended chemical controls: {chem}.")
        if org:
            seeds.append(f"Organic or cultural measures include: {org}.")
        if prev:
            seeds.append(f"Preventive strategies: {prev}.")

    # Fallback seeds when minimal info
    if not seeds:
        seeds = [
            f"{disease or 'This disease'} affects leaves and can lower plant vigor.",
            "Typical signs are spots, lesions, and discoloration.",
            "Management combines cultural practices, monitoring, and targeted treatments.",
        ]

    # Templates to rephrase and expand the seeds into coherent paragraphs
    templates = [
        "{s}",
        "In many cases, {short} is observed, which can lead to reduced vigor and yield.",
        "Growers should note that {short} often indicates progressing infection.",
        "Timely action addressing {short} helps limit spread and protects yield.",
        "Prevention and early detection are key: {short} should prompt monitoring and control measures.",
        "When environmental conditions are favorable, {short} may worsen quickly.",
        "Consult local extension services for tailored advice when {short} appears widespread.",
    ]

    # Build paragraph without verbatim repeats by cycling seeds and templates
    words: list = []
    used_phrases = set()
    i = 0
    while len(words) < 200:
        seed = seeds[i % len(seeds)]
        # make a short preview for template insertion
        short = " ".join(re.split(r"[\n\.;,]+", seed)[0].split()[:12]).rstrip('.,')
        tpl = templates[i % len(templates)]
        sentence = tpl.format(s=seed, short=short)

        # Normalize whitespace and trim
        sentence = re.sub(r"\s+", " ", sentence).strip()

        # Avoid adding an identical sentence twice in a row
        if len(words) > 0:
            last_fragment = " ".join(words[-20:])
            if sentence in used_phrases or sentence.strip() in last_fragment:
                # create a variant phrasing
                sentence = f"Notably, {short}."

        used_phrases.add(sentence)
        words.extend(sentence.split())
        i += 1

    words = words[:200]
    out = " ".join(words)
    out = out if out.endswith('.') else out + '.'
    # Post-process to remove obvious repetitions
    out = _deduplicate_text(out)
    return out


def _expand_to_n_words(text: str, disease: str = "", n: int = 200) -> str:
    """Expand a short text to approximately `n` words by repeating meaningful chunks.

    Keeps readability by choosing a sensible chunk to repeat and trimming to exactly n words.
    """
    import re

    if not text or not isinstance(text, str):
        text = f"{disease} description not available."

    # Quick return when already long enough
    words = text.split()
    if len(words) >= n:
        out = " ".join(words[:n])
        return out if out.endswith(".") else out + "."

    # Split into sentences to avoid repeating the exact same fragment
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if not sentences:
        sentences = [text]

    connectors = [
        "Additionally,",
        "Moreover,",
        "In particular,",
        "Consequently,",
        "This often leads to",
        "Importantly,",
        "Furthermore,",
        "As a result,",
    ]

    out_words: list = []
    i = 0
    # Start with original text words
    out_words.extend(words)

    # Cycle through sentences with varied connectors and occasional disease mention
    while len(out_words) < n:
        sent = sentences[i % len(sentences)]
        # avoid repeating the exact same sentence fragment verbatim
        conn = connectors[i % len(connectors)] if i % len(connectors) != 0 else ""
        prefix = f"For {disease}," if (i % 5 == 0 and disease) else ""

        # build a candidate chunk
        candidate = " ".join(p for p in [prefix, conn, sent] if p).strip()

        # If candidate is identical to the last added chunk, create a short variant
        if len(out_words) > 0:
            # inspect last added words to detect repetition
            last_chunk = " ".join(out_words[-min(len(words), 40):]).strip()
        else:
            last_chunk = ""

        if candidate == last_chunk or candidate == sent:
            # create a gentle rephrase to avoid exact repetition
            short_preview = " ".join(sent.split()[:12]).rstrip('.,')
            variant = f"This typically presents as {short_preview}."
            chunk_text = variant
        else:
            chunk_text = candidate

        # Fallback: ensure we always add something useful
        if not chunk_text:
            chunk_text = sent

        out_words.extend(chunk_text.split())
        i += 1

    out_words = out_words[:n]
    out = " ".join(out_words)
    return out if out.endswith(".") else out + "."


def _synthesize_section_detail(short_text: str, long_text: str, disease: str = "", n: int = 200) -> str:
    """Create a varied, readable section detail of approximately `n` words.

    Uses the short original text as the seed for generating paraphrase-style
    sentences rather than repeating the long expanded block verbatim.
    """
    import re

    seed = short_text if short_text and isinstance(short_text, str) and short_text.strip() else long_text
    if not seed or not isinstance(seed, str):
        seed = "Information not available."

    words = seed.split()
    if len(words) >= n:
        out = " ".join(words[:n])
        return out if out.endswith('.') else out + '.'

    # break seed into short fragments
    parts = [s.strip() for s in re.split(r"[\n\.;]+|,", seed) if s.strip()]
    if not parts:
        parts = [seed]

    connectors = ["Additionally,", "Moreover,", "Importantly,", "Consequently,", "Furthermore,"]
    templates = [
        "{p}.",
        "This typically appears as {short}.",
        "Affected plants often show {short}, which may lead to reduced vigor.",
        "Management should focus on addressing {short} to limit spread.",
        "Early detection of {short} enables timely intervention.",
    ]

    out_words = []
    i = 0
    used = set()
    while len(out_words) < n:
        part = parts[i % len(parts)]
        short_preview = " ".join(part.split()[:8]).rstrip('.,')
        tpl = templates[i % len(templates)]
        conn = connectors[i % len(connectors)] if i % 2 == 0 else ""
        cand = tpl.format(p=part, short=short_preview)
        chunk = " ".join(p for p in [conn, cand] if p).strip()
        # avoid exact duplicates
        if chunk in used:
            # fallback to a slightly different phrasing
            chunk = f"{part}. " + f"Notably, {short_preview}."
        used.add(chunk)
        out_words.extend(chunk.split())
        i += 1

    out_words = out_words[:n]
    out = " ".join(out_words)
    out = out if out.endswith('.') else out + '.'
    # Reduce repetition within synthesized section detail as well
    out = _deduplicate_text(out)
    return out


def _deduplicate_text(text: str) -> str:
    """Remove repeated adjacent sentences and repeated words/phrases.

    This is a conservative, deterministic post-processing step to reduce
    obvious verbatim repetition without changing meaning significantly.
    """
    if not text or not isinstance(text, str):
        return text
    # collapse long runs of the same word (e.g., "fruit fruit fruit")
    text = re.sub(r"\b(\w+)(?:\s+\1){2,}\b", r"\1", text, flags=re.I)

    # split into sentences
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if not sentences:
        return text

    out_sents = []
    seen = set()
    for s in sentences:
        norm = re.sub(r"[^a-z0-9 ]", "", s.lower())
        norm = re.sub(r"\s+", " ", norm).strip()
        if not norm:
            continue
        # skip exact duplicates
        if norm in seen:
            continue
        # skip if duplicate or substring of last sentence
        if out_sents:
            last_norm = re.sub(r"[^a-z0-9 ]", "", out_sents[-1].lower())
            last_norm = re.sub(r"\s+", " ", last_norm).strip()
            if norm == last_norm or norm in last_norm or last_norm in norm:
                continue
        seen.add(norm)
        out_sents.append(s)

    final = " ".join(out_sents)
    # final cleanup of small repeated token runs
    final = re.sub(r"\b(\w+)(?:\s+\1){1,}\b", r"\1", final, flags=re.I)
    return final


def _deep_copy(d: dict) -> dict:
    """Simple deep copy for dicts containing only str/dict values."""
    out = {}
    for k, v in d.items():
        out[k] = _deep_copy(v) if isinstance(v, dict) else v
    return out


def _ensure_min_words_in_rec(rec: dict, min_words: int = 200) -> None:
    """Ensure the recommendation `rec` has at least `min_words` in key text fields.

    This mutates `rec` in-place. Fields checked: visible_symptoms, probable_cause,
    prevention, treatment.chemical_control, treatment.organic_control, disease_description.
    """
    if not isinstance(rec, dict):
        return

    def _ensure_field(key: str, fallback: str):
        val = rec.get(key, "")
        if not isinstance(val, str) or len(val.split()) < min_words:
            rec[key] = _expand_to_n_words(val if val else fallback, rec.get("disease_name", ""), n=min_words)

    disease = rec.get("disease_name", "")
    _ensure_field("visible_symptoms", "Symptoms include leaf spots, discoloration, and necrosis.")
    _ensure_field("probable_cause", "It is commonly caused by fungal, bacterial, or viral pathogens under conducive environmental conditions.")
    _ensure_field("prevention", "Preventive measures include cultural practices, resistant varieties, and timely chemical or biological controls.")
    # disease_description may already be present
    if not isinstance(rec.get("disease_description", ""), str) or len(rec.get("disease_description", "").split()) < min_words:
        rec["disease_description"] = _expand_to_n_words(rec.get("disease_description", ""), disease, n=min_words)

    if isinstance(rec.get("treatment"), dict):
        tc = rec["treatment"].get("chemical_control", "")
        if not isinstance(tc, str) or len(tc.split()) < min_words:
            rec["treatment"]["chemical_control"] = _expand_to_n_words(tc if tc else "Apply appropriate fungicide or pesticide as recommended by local agricultural authorities.", disease, n=min_words)
        to = rec["treatment"].get("organic_control", "")
        if not isinstance(to, str) or len(to.split()) < min_words:
            rec["treatment"]["organic_control"] = _expand_to_n_words(to if to else "Remove affected plant parts, improve air circulation, and use neem-based products or biological controls.", disease, n=min_words)

    # Post-process treatment fields to remove repetition and improve formatting
    try:
        if isinstance(rec.get("treatment"), dict):
            for key in ("chemical_control", "organic_control"):
                txt = rec["treatment"].get(key, "")
                if not txt or not isinstance(txt, str):
                    continue
                # First, deduplicate obvious verbatim repeats
                txt = _deduplicate_text(txt)

                # Split into sentences and remove near-duplicates while preserving order
                sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', txt) if s.strip()]
                seen = set()
                unique_sents = []
                for s in sents:
                    norm = re.sub(r"[^a-z0-9 ]", "", s.lower())
                    norm = re.sub(r"\s+", " ", norm).strip()
                    if not norm:
                        continue
                    if norm in seen:
                        continue
                    seen.add(norm)
                    unique_sents.append(s)

                # Group sentences into short paragraphs (~3 sentences per paragraph)
                paragraphs = []
                for i in range(0, len(unique_sents), 3):
                    para = " ".join(unique_sents[i:i+3]).strip()
                    if para:
                        paragraphs.append(para)

                # If nothing remained after cleaning, keep original deduped text
                final = "\n\n".join(paragraphs) if paragraphs else txt

                # Ensure final ends with a period
                if final and not final.endswith("."):
                    final = final + "."

                rec["treatment"][key] = final
    except Exception:
        pass


# -- Module-level singleton --
recommendation_engine = RecommendationEngine()
