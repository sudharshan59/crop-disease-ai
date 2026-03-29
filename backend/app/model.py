"""
CNN Disease Classification Model module.

Handles loading, managing, and running inference with the ResNet18-based
plant disease classifier. Supports both standard PyTorch (.pth) and
TorchScript (.pt) model formats.
"""

import logging
import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path
from typing import Tuple, Optional

from app.config import settings, NUM_CLASSES, DISEASE_CLASSES, DISEASE_DISPLAY_NAMES

logger = logging.getLogger(__name__)


def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """Build an EfficientNet-B0 model with a modified classifier head.

    Uses torchvision EfficientNet-B0 for a good accuracy/size tradeoff on
    edge devices. Default uses ImageNet pretrained weights.
    """
    try:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
    except Exception:
        # Fallback: if torchvision older version lacks efficientnet, use resnet18
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)

    # Replace classifier head
    if hasattr(model, "classifier"):
        in_features = model.classifier[1].in_features if isinstance(model.classifier, nn.Sequential) else model.classifier.in_features
        model.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features, num_classes))
    else:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    return model


class DiseaseClassifier:
    """Wrapper for the plant disease CNN classifier.

    Manages model loading (PyTorch weights or TorchScript), device placement,
    and inference with confidence scoring.
    """

    def __init__(self):
        self.model: Optional[nn.Module] = None
        self.device: str = settings.get_device()
        self.is_torchscript: bool = False
        self.is_loaded: bool = False

    def load(self) -> bool:
        """Load the CNN model from disk.

        Tries TorchScript first, then falls back to standard .pth weights.
        If no model file exists, builds a model with random weights for demo.

        Returns:
            True if model loaded successfully.
        """
        torchscript_path = Path(settings.TORCHSCRIPT_PATH)
        weights_path = Path(settings.MODEL_PATH)

        try:
            if torchscript_path.exists():
                # Prefer TorchScript for edge inference
                logger.info(f"Loading TorchScript model from {torchscript_path}")
                self.model = torch.jit.load(str(torchscript_path), map_location=self.device)
                self.is_torchscript = True
                logger.info("TorchScript model loaded successfully.")

            elif weights_path.exists():
                # Load standard PyTorch weights
                logger.info(f"Loading PyTorch model weights from {weights_path}")
                self.model = build_resnet18(num_classes=NUM_CLASSES, pretrained=False)
                state_dict = torch.load(str(weights_path), map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)
                logger.info("PyTorch model weights loaded successfully.")

            else:
                # No trained model found — use ImageNet-pretrained backbone for demo
                logger.warning(
                    f"No model file found at {weights_path} or {torchscript_path}. "
                    "Building model with ImageNet-pretrained backbone (demo mode). "
                    "Predictions will vary by image but won't be accurate. "
                    "Run train.py to train a proper model."
                )
                # build_model provides EfficientNet-B0 or fallback ResNet18
                self.model = build_model(num_classes=NUM_CLASSES, pretrained=True)

            # Move to device and set to eval mode
            self.model = self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            logger.info(f"Model ready on device: {self.device}")
            return True

        except Exception as e:
            logger.error(f"Failed to load CNN model: {e}", exc_info=True)
            self.is_loaded = False
            return False

    def predict(self, tensor: torch.Tensor) -> Tuple[int, str, str, float, list]:
        """Run inference on a preprocessed image tensor.

        Args:
            tensor: Preprocessed image tensor of shape (1, 3, 224, 224).

        Returns:
            Tuple of:
            - class_index (int): Predicted class index.
            - class_label (str): Raw class label string.
            - display_name (str): Human-readable disease name.
            - confidence (float): Confidence score (0–1) for the predicted class.
            - probabilities (list): Full probability distribution across all classes.

        Raises:
            RuntimeError: If the model is not loaded.
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("CNN model is not loaded. Call load() first.")

        # Move tensor to model device
        tensor = tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)

            class_index = predicted_idx.item()
            confidence_score = round(confidence.item(), 4)
            prob_list = probabilities.squeeze().tolist()

            # Map to class labels
            class_label = DISEASE_CLASSES[class_index]
            display_name = DISEASE_DISPLAY_NAMES.get(class_label, class_label)

        return class_index, class_label, display_name, confidence_score, prob_list

    def is_healthy(self, class_label: str) -> bool:
        """Check if the prediction indicates a healthy plant.

        Args:
            class_label: Raw class label from prediction.

        Returns:
            True if the plant is classified as healthy.
        """
        return class_label == "Healthy"

    def get_status(self) -> dict:
        """Get model status information.

        Returns:
            Dictionary with model status details.
        """
        # Determine a friendly model name based on what was loaded
        try:
            from app.config import settings
            model_name = None
            if self.is_torchscript and settings.TORCHSCRIPT_PATH:
                model_name = Path(settings.TORCHSCRIPT_PATH).name
            elif settings.MODEL_PATH and Path(settings.MODEL_PATH).exists():
                model_name = Path(settings.MODEL_PATH).name
            else:
                # Fallback to the backbone name we built (EfficientNet or ResNet)
                backbone = "EfficientNet-B0"
                try:
                    import torchvision.models as models
                    if isinstance(self.model, models.ResNet):
                        backbone = "ResNet18"
                except Exception:
                    pass
                model_name = f"{backbone} (fallback)"
        except Exception:
            model_name = "unknown"

        return {
            "loaded": self.is_loaded,
            "device": self.device,
            "torchscript": self.is_torchscript,
            "model_name": model_name,
            "num_classes": NUM_CLASSES,
            "classes": DISEASE_CLASSES,
        }


# ─── Module-level Singleton ──────────────────────────────────────────────────
classifier = DiseaseClassifier()

def _load_second_model(path: str, device: str) -> Optional[nn.Module]:
    """Helper to load an alternate model from a given path.

    Returns the loaded model on success, or None on failure.
    """
    try:
        p = Path(path)
        if not p.exists():
            return None
        # Try TorchScript first if file is a .pt
        if p.suffix == ".pt":
            m = torch.jit.load(str(p), map_location=device)
            m.eval()
            return m.to(device)
        else:
            # Assume it's a state_dict compatible with our ResNet18 architecture
            m = build_resnet18(num_classes=NUM_CLASSES, pretrained=False)
            state_dict = torch.load(str(p), map_location=device, weights_only=True)
            m.load_state_dict(state_dict)
            m = m.to(device)
            m.eval()
            return m
    except Exception:
        return None


def _choose_higher(pred1, pred2):
    """Return the prediction tuple with higher confidence between two results.

    Each pred is (class_index, class_label, display_name, confidence, probabilities).
    """
    if pred2 is None:
        return pred1, None
    if pred1 is None:
        return pred2, None
    return (pred1, pred2) if pred1[3] >= pred2[3] else (pred2, pred1)


def dual_predict(tensor: torch.Tensor) -> dict:
    """Run inference with the primary loaded model and an alternate model (if present).

    The alternate model path searched (in order): `settings.TORCHSCRIPT_PATH`,
    `settings.MODEL_PATH` (if different from the loaded model path), and
    `models/alt_plant_disease_model.pth` in the models folder.

    Returns a dict containing both predictions and a `winner` field with the
    higher-confidence result.
    """
    # Primary prediction
    try:
        primary = classifier.predict(tensor)
    except Exception as e:
        raise

    # Attempt to locate alternate model files
    alt_paths = []
    try:
        # prefer TorchScript alternate
        if settings.TORCHSCRIPT_PATH:
            alt_paths.append(settings.TORCHSCRIPT_PATH)
        # alternate weights path (avoid same as primary MODEL_PATH)
        if settings.MODEL_PATH and settings.MODEL_PATH not in alt_paths:
            alt_paths.append(settings.MODEL_PATH)
        # fallback alt path in models folder
        alt_paths.append(str(Path(settings.MODELS_DIR) / "alt_plant_disease_model.pth"))
    except Exception:
        alt_paths = []

    alt_pred = None
    for p in alt_paths:
        # skip if identical to primary model file
        try:
            if p and Path(p).exists():
                # load alternate model temporarily
                alt_model = _load_second_model(p, classifier.device)
                if alt_model is None:
                    continue
                # run inference with alt_model
                with torch.no_grad():
                    t = tensor.to(classifier.device)
                    outputs = alt_model(t)
                    probs = torch.softmax(outputs, dim=1)
                    conf, idx = torch.max(probs, dim=1)
                    idx_i = idx.item()
                    conf_f = round(conf.item(), 4)
                    prob_list = probs.squeeze().tolist()
                    class_label = DISEASE_CLASSES[idx_i]
                    display_name = DISEASE_DISPLAY_NAMES.get(class_label, class_label)
                    alt_pred = (idx_i, class_label, display_name, conf_f, prob_list)
                    break
        except Exception:
            continue

    # Decide winner
    winner, loser = _choose_higher(primary, alt_pred)

    return {
        "primary": {
            "class_index": primary[0],
            "class_label": primary[1],
            "display_name": primary[2],
            "confidence": primary[3],
            "probabilities": primary[4],
        },
        "alternate": None if alt_pred is None else {
            "class_index": alt_pred[0],
            "class_label": alt_pred[1],
            "display_name": alt_pred[2],
            "confidence": alt_pred[3],
            "probabilities": alt_pred[4],
        },
        "winner": {
            "class_index": winner[0],
            "class_label": winner[1],
            "display_name": winner[2],
            "confidence": winner[3],
        },
        "loser": None if loser is None else {
            "class_index": loser[0],
            "class_label": loser[1],
            "display_name": loser[2],
            "confidence": loser[3],
        }
    }
