# Model Details

## 1. CNN Disease Classifier — ResNet18

### Architecture

- **Base Model:** ResNet18 (He et al., 2015)
- **Pretrained Weights:** ImageNet-1K (torchvision `IMAGENET1K_V1`)
- **License:** MIT License (torchvision)
- **Modification:** Final fully connected layer replaced: `Linear(512, 1000)` → `Linear(512, 16)`
- **Input:** RGB image tensor `[B, 3, 224, 224]`
- **Output:** 16-class logits → Softmax probabilities

### Disease Classes (16)

| Index | Class Label | Crop | Disease |
|-------|-------------|------|---------|
| 0 | Apple___Apple_scab | Apple | Apple Scab |
| 1 | Corn_(maize)___Cercospora_leaf_spot | Corn | Gray Leaf Spot |
| 2 | Corn_(maize)___Common_rust | Corn | Common Rust |
| 3 | Corn_(maize)___Northern_Leaf_Blight | Corn | Northern Leaf Blight |
| 4 | Grape___Black_rot | Grape | Black Rot |
| 5 | Grape___Esca_(Black_Measles) | Grape | Black Measles (Esca) |
| 6 | Potato___Early_blight | Potato | Early Blight |
| 7 | Potato___Late_blight | Potato | Late Blight |
| 8 | Tomato___Bacterial_spot | Tomato | Bacterial Spot |
| 9 | Tomato___Early_blight | Tomato | Early Blight |
| 10 | Tomato___Late_blight | Tomato | Late Blight |
| 11 | Tomato___Leaf_Mold | Tomato | Leaf Mold |
| 12 | Tomato___Septoria_leaf_spot | Tomato | Septoria Leaf Spot |
| 13 | Tomato___Target_Spot | Tomato | Target Spot |
| 14 | Tomato___Yellow_Leaf_Curl_Virus | Tomato | Yellow Leaf Curl Virus |
| 15 | Healthy | — | Healthy (No Disease) |

### Training Configuration

- **Dataset:** PlantVillage (subset of 16 classes, ~25,000 images)
- **Split:** 80% train / 10% validation / 10% test
- **Optimizer:** Adam (lr=1e-4, weight_decay=1e-4)
- **Loss Function:** CrossEntropyLoss
- **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=3)
- **Early Stopping:** Patience = 5 epochs
- **Batch Size:** 32
- **Epochs:** Up to 30 (typical convergence: 15-20)

### Data Augmentation

| Augmentation | Parameters |
|-------------|-----------|
| RandomResizedCrop | 224×224, scale=(0.8, 1.0) |
| RandomHorizontalFlip | p=0.5 |
| RandomRotation | ±15° |
| ColorJitter | brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05 |
| Normalize | ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |

### Fine-tuning Strategy

1. **Phase 1 (Epochs 1-5):** Freeze all layers except `layer4` and `fc`. Train with lr=1e-4.
2. **Phase 2 (Epochs 6+):** Unfreeze all layers. Reduce lr to 1e-5. Full fine-tuning.

### Expected Performance

- **PlantVillage Test Accuracy:** 95-99%
- **Inference Time (CPU):** ~20-50ms per image
- **Inference Time (GPU):** ~5-10ms per image
- **Model Size:** ~44.7 MB (fp32 .pth), ~11 MB (TorchScript)

### TorchScript Export

The trained model is exported using `torch.jit.trace` for edge deployment:

```python
dummy_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save("plant_disease_model_scripted.pt")
```

This enables:
- Framework-free inference (no Python/torchvision dependency)
- Mobile deployment (Android/iOS via LibTorch)
- Optimized execution with TorchScript JIT compiler

---

## 2. LLM Recommendation Engine — Phi-2

### Primary Model: Microsoft Phi-2

- **Model:** `microsoft/phi-2`
- **Parameters:** 2.7 billion
- **License:** MIT License
- **Architecture:** Transformer decoder-only
- **Context Window:** 2048 tokens
- **Training Data:** Textbooks, web data, synthetic data

### Alternative Model: Mistral-7B-Instruct

- **Model:** `mistralai/Mistral-7B-Instruct-v0.2`
- **Parameters:** 7.24 billion
- **License:** Apache 2.0
- **Use Case:** When GPU with ≥8GB VRAM is available

### Quantization

#### GGUF Quantization (Default — CPU)

- **Format:** GGUF (GPT-Generated Unified Format)
- **Quantization Level:** Q4_K_M (4-bit, medium quality)
- **Library:** llama-cpp-python
- **Model File:** `phi-2.Q4_K_M.gguf` (~1.8 GB)
- **RAM Usage:** ~2-3 GB
- **CPU Inference Speed:** ~2-5 seconds/token

**How GGUF Quantization Works:**

1. Original FP16 weights (5.4 GB) are quantized to 4-bit integers
2. Uses k-quant scheme: groups of weights share scale factors
3. Q4_K_M retains important layers at higher precision (K = key layers)
4. Reduces memory by ~3x while maintaining >95% of original quality
5. Enables CPU inference through optimized GGML tensor operations

#### bitsandbytes Quantization (GPU Alternative)

- **Library:** bitsandbytes
- **Quantization:** NF4 (normalized float 4-bit)
- **Double Quantization:** Enabled (quantizes the quantization constants)
- **Compute dtype:** float16
- **VRAM Usage:** ~3-4 GB for phi-2, ~5-6 GB for Mistral-7B
- **Configuration:**

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
```

**NF4 Quantization:**
- Information-theoretically optimal data type for normally distributed weights
- Double quantization applies 8-bit quantization to the FP32 scaling constants
- Reduces memory per parameter from 16 bits to ~4.5 bits effectively
- Near-zero quality loss on downstream tasks

### Prompt Engineering

The system uses a structured prompt template that:

1. Sets the AI role as "expert agricultural pathologist and extension officer"
2. Provides diagnosis context (disease, confidence, severity, region)
3. Requests output in strict JSON format with 4 fields
4. Adjusts treatment intensity based on severity level
5. Optionally includes regional context for localized recommendations

### Fallback System

When LLM is unavailable, the system falls back to a hardcoded expert recommendation dictionary containing:
- 16 pre-written expert recommendations (one per disease class)
- Severity-aware treatment notes appended dynamically
- Covers: description, causes, treatment (cultural/biological/chemical), prevention
