"""
OpenCV-based image preprocessing pipeline for leaf disease detection.

Pipeline Steps:
1. Decode raw image bytes to BGR numpy array
2. Resize to 224×224
3. Apply Gaussian blur for noise reduction
4. Color normalization (BGR → RGB, ImageNet stats)
5. Optional green-channel segmentation for leaf isolation
6. Convert to PyTorch tensor (batch-ready)
"""

import cv2
import numpy as np
import torch
from typing import Tuple, Optional

from app.config import IMAGE_SIZE, GAUSSIAN_BLUR_KERNEL, IMAGENET_MEAN, IMAGENET_STD


def decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes into a BGR numpy array.

    Args:
        image_bytes: Raw bytes from uploaded image file.

    Returns:
        BGR image as numpy array (H, W, 3).

    Raises:
        ValueError: If the image cannot be decoded.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image. Ensure the file is a valid image (JPEG/PNG).")
    return image


def resize_image(image: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Resize image to target square dimensions.

    Args:
        image: Input BGR image array.
        size: Target dimension (default 224).

    Returns:
        Resized image (size × size × 3).
    """
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)


def apply_gaussian_blur(image: np.ndarray, kernel: Tuple[int, int] = GAUSSIAN_BLUR_KERNEL) -> np.ndarray:
    """Apply Gaussian blur for noise reduction.

    Args:
        image: Input image array.
        kernel: Gaussian kernel size (default 5×5).

    Returns:
        Blurred image.
    """
    return cv2.GaussianBlur(image, kernel, 0)


def normalize_colors(image: np.ndarray) -> np.ndarray:
    """Convert BGR to RGB and normalize pixel values to [0, 1].

    Args:
        image: Input BGR image array.

    Returns:
        RGB normalized image as float32 array with values in [0, 1].
    """
    # BGR → RGB conversion
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Normalize to [0, 1]
    normalized = rgb_image.astype(np.float32) / 255.0
    return normalized


def segment_leaf(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Perform green-channel based leaf segmentation.

    Creates a mask isolating the leaf region from the background using
    HSV color space thresholding focused on green/vegetation.

    Args:
        image: Input BGR image array.

    Returns:
        Tuple of (segmented_image, binary_mask).
        - segmented_image: Original image with non-leaf areas set to black.
        - binary_mask: Binary mask (0/255) of detected leaf region.
    """
    # Convert to HSV for better color-based segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for green vegetation (broad range to capture diseased leaves too)
    lower_green = np.array([15, 20, 20])
    upper_green = np.array([95, 255, 255])

    # Also capture brown/yellow diseased tissue
    lower_brown = np.array([5, 20, 20])
    upper_brown = np.array([25, 255, 255])

    # Create combined mask
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
    mask = cv2.bitwise_or(mask_green, mask_brown)

    # Morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Apply mask to original image
    segmented = cv2.bitwise_and(image, image, mask=mask)

    return segmented, mask


def compute_diseased_area_ratio(mask: np.ndarray, original_bgr: np.ndarray) -> float:
    """Compute the ratio of discolored/diseased area on the leaf.

    Uses color analysis within the leaf mask to estimate what percentage
    of the leaf area shows discoloration (brown, yellow, spots).

    Args:
        mask: Binary leaf segmentation mask.
        original_bgr: Original BGR image.

    Returns:
        Float between 0.0 and 1.0 representing the disease coverage ratio.
    """
    if mask is None or np.sum(mask) == 0:
        return 0.0

    # Convert to HSV within the leaf region
    hsv = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2HSV)

    # Healthy green range (narrower than segmentation range)
    lower_healthy = np.array([25, 40, 40])
    upper_healthy = np.array([85, 255, 255])
    healthy_mask = cv2.inRange(hsv, lower_healthy, upper_healthy)

    # Only consider pixels within the leaf
    leaf_pixels = np.sum(mask > 0)
    healthy_pixels = np.sum(cv2.bitwise_and(healthy_mask, mask) > 0)

    if leaf_pixels == 0:
        return 0.0

    # Diseased ratio = 1 - healthy ratio
    healthy_ratio = healthy_pixels / leaf_pixels
    diseased_ratio = max(0.0, min(1.0, 1.0 - healthy_ratio))

    return diseased_ratio


def image_to_tensor(
    image: np.ndarray,
    mean: list = IMAGENET_MEAN,
    std: list = IMAGENET_STD,
) -> torch.Tensor:
    """Convert a normalized RGB image to a batch-ready PyTorch tensor.

    Applies ImageNet mean/std normalization, then converts to tensor
    with shape [1, 3, 224, 224].

    Args:
        image: RGB float32 image normalized to [0, 1], shape (H, W, 3).
        mean: Per-channel mean for normalization.
        std: Per-channel std for normalization.

    Returns:
        Batch-ready tensor of shape (1, 3, H, W).
    """
    # Apply ImageNet normalization per channel
    mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    normalized = (image - mean_arr) / std_arr

    # HWC → CHW and add batch dimension
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
    return tensor


def preprocess_image(
    image_bytes: bytes,
    apply_segmentation: bool = False,
) -> Tuple[torch.Tensor, Optional[float]]:
    """Full preprocessing pipeline: bytes → batch-ready tensor.

    Steps:
    1. Decode image bytes → BGR array
    2. Resize to 224×224
    3. Apply Gaussian blur
    4. (Optional) Segment leaf region
    5. Normalize colors (BGR→RGB, scale to [0,1])
    6. Apply ImageNet normalization
    7. Convert to tensor [1, 3, 224, 224]

    Args:
        image_bytes: Raw bytes from uploaded image file.
        apply_segmentation: Whether to apply leaf segmentation.

    Returns:
        Tuple of:
        - Preprocessed tensor ready for model input (1, 3, 224, 224).
        - Diseased area ratio (float 0-1) if segmentation is applied, else None.
    """
    # Step 1: Decode
    bgr_image = decode_image(image_bytes)

    # Step 2: Resize
    resized = resize_image(bgr_image, IMAGE_SIZE)

    # Step 3: Gaussian blur for noise reduction
    blurred = apply_gaussian_blur(resized)

    # Step 4: Optional segmentation
    diseased_ratio: Optional[float] = None
    if apply_segmentation:
        segmented, mask = segment_leaf(blurred)
        diseased_ratio = compute_diseased_area_ratio(mask, blurred)
        processing_image = segmented
    else:
        processing_image = blurred

    # Step 5: Color normalization (BGR → RGB, scale to [0, 1])
    normalized = normalize_colors(processing_image)

    # Step 6-7: ImageNet normalization and tensor conversion
    tensor = image_to_tensor(normalized)

    return tensor, diseased_ratio
