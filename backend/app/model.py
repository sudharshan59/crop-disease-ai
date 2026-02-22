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


def build_resnet18(num_classes: int = NUM_CLASSES, pretrained: bool = False) -> nn.Module:
    """Build a ResNet18 model with a modified final FC layer.

    Args:
        num_classes: Number of output classes (default: 16).
        pretrained: Whether to load ImageNet pretrained weights.

    Returns:
        Modified ResNet18 model.
    """
    if pretrained:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
    else:
        weights = None

    model = models.resnet18(weights=weights)
    # Replace final fully connected layer for our classification task
    in_features = model.fc.in_features  # 512 for ResNet18
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
                self.model = build_resnet18(num_classes=NUM_CLASSES, pretrained=True)

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
        return {
            "loaded": self.is_loaded,
            "device": self.device,
            "torchscript": self.is_torchscript,
            "num_classes": NUM_CLASSES,
            "classes": DISEASE_CLASSES,
        }


# ─── Module-level Singleton ──────────────────────────────────────────────────
classifier = DiseaseClassifier()
