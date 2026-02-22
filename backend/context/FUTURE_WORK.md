# Future Work

This document outlines the planned roadmap for extending the Crop Disease Detection & Treatment Recommendation System, aligned with the research paper's future work section.

---

## 1. Multi-Dataset Fusion Support

### Current State
- Trained on PlantVillage dataset (lab-controlled images)
- 16 classes across 5 crop types

### Planned Expansion
- **Integrate field-captured datasets** (IP102, PlantDoc, custom field images)
- **Domain adaptation** techniques to bridge the lab → real-world gap:
  - Style transfer augmentation (CycleGAN)
  - Unsupervised domain adaptation (DANN)
  - Mixed training with stratified sampling
- **Multi-dataset DataLoader** that samples from multiple sources per batch
- **Dataset versioning** with DVC (Data Version Control) for reproducibility

### Implementation Approach
```
data/
├── plantvillage/       # Lab images (existing)
├── plantdoc/           # Real-world document images
├── field_images/       # Farmer-submitted images
└── dataset_config.yaml # Sampling weights and splits
```

---

## 2. Severity-Based Recommendation Adjustment

### Current State
- Severity (mild/moderate/severe) is passed to LLM prompt
- Fallback dictionary appends severity-specific notes

### Planned Enhancement
- **Graduated treatment protocols** per severity level:
  - Mild → Cultural methods only
  - Moderate → Cultural + biological
  - Severe → Cultural + biological + chemical
- **Dosage adjustment** in chemical recommendations based on severity
- **Economic impact estimation** (yield loss prediction per severity)
- **Follow-up scheduling** recommendations based on severity progression
- **Multi-image temporal tracking** — compare severity across visits

---

## 3. Edge Inference Mode (TorchScript / ONNX)

### Current State
- TorchScript export implemented in training pipeline
- Models can be loaded from `.pt` files

### Planned Enhancement
- **ONNX Runtime** export for cross-platform deployment
- **TensorFlow Lite** conversion for Android mobile apps
- **Core ML** conversion for iOS deployment
- **Quantized edge models** (INT8 quantization for mobile):
  ```python
  model_int8 = torch.quantization.quantize_dynamic(
      model, {nn.Linear}, dtype=torch.qint8
  )
  ```
- **Offline-first mobile app** with on-device inference
- **Progressive enhancement**: edge CNN + cloud LLM backup
- **WebAssembly (WASM)** compilation for browser-based inference

### Target Platforms
| Platform | Format | Size | Latency |
|----------|--------|------|---------|
| Cloud Server | PyTorch | ~45 MB | <50ms |
| Edge Server | TorchScript | ~11 MB | <100ms |
| Android | TFLite | ~5 MB | <200ms |
| iOS | Core ML | ~5 MB | <200ms |
| Browser | ONNX.js/WASM | ~6 MB | <500ms |

---

## 4. Personalization & Region-Based Suggestions

### Current State
- Optional `region` parameter in API
- Region context included in LLM prompt

### Planned Enhancement
- **Regional crop calendars** — season-aware recommendations
- **Local pesticide regulations** — region-specific chemical recommendations
- **Language localization** — LLM generates recommendations in local language
- **Local variety recommendations** — suggest region-specific resistant cultivars
- **Weather integration** — real-time weather-aware disease risk assessment
- **Farmer profile system**:
  - Crop history
  - Farm size and type (organic/conventional)
  - Previous disease occurrences
  - Available equipment/resources

### Data Sources
- Regional agricultural extension databases
- FAO Crop Calendar data
- National pesticide registries
- OpenWeatherMap API for weather integration

---

## 5. Prompt Logging & Refinement Pipeline

### Current State
- `LOG_PROMPTS=True` enables prompt logging in memory
- Accessible via `recommendation_engine.get_prompt_log()`

### Planned Enhancement
- **Persistent prompt logging** to SQLite/file
- **A/B testing framework** for prompt variants
- **Farmer feedback collection** — rate recommendation quality
- **Automated prompt optimization**:
  - Log prompt → response → farmer rating
  - Use feedback to refine prompt templates
  - DSPy-style prompt compilation
- **RAG (Retrieval Augmented Generation)** pipeline:
  - Vector database of agricultural knowledge
  - Retrieve relevant context per disease/crop/region
  - Inject into prompt for more accurate recommendations
- **Few-shot learning** from successful recommendations

### Prompt Refinement Pipeline
```
                    ┌─────────────┐
User Feedback ────▶│  Feedback DB │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
Historical Data ──▶│   Analyzer   │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  New Prompt  │
                    │  Templates   │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   A/B Test   │──▶ Best Variant Promoted
                    └─────────────┘
```

---

## 6. Modular Crop Expansion System

### Current State
- Fixed 16-class model covering 5 crops
- Adding new crops requires full retraining

### Planned Enhancement
- **Hierarchical classification**:
  1. Stage 1: Crop identification (what plant?)
  2. Stage 2: Crop-specific disease model (what disease?)
- **Dynamic model registry**:
  ```
  models/
  ├── crop_identifier.pt          # Crop detection (common)
  ├── tomato_diseases.pt          # 10+ tomato diseases
  ├── potato_diseases.pt          # 5+ potato diseases
  ├── corn_diseases.pt            # 4+ corn diseases
  ├── rice_diseases.pt            # Future: rice support
  └── model_registry.json         # Model metadata & routing
  ```
- **Hot-swappable models** — add new crop models without system restart
- **Transfer learning pipeline** — rapidly fine-tune for new crops:
  1. Start from shared backbone
  2. Fine-tune crop-specific heads with 500+ images
  3. Validate on held-out test set
  4. Register in model registry
  5. Deploy without retraining existing models
- **Community contribution system** — farmers submit labeled images for new crops
- **Active learning** — system identifies uncertain predictions and requests labels

### Priority Crops for Expansion
1. Rice (blast, sheath blight, bacterial leaf blight)
2. Wheat (rust, powdery mildew, Septoria)
3. Soybean (rust, downy mildew, mosaic virus)
4. Cotton (bollworm damage, leaf curl virus)
5. Cassava (mosaic disease, bacterial blight)

---

## 7. Additional Future Enhancements

### Multi-Modal Input
- **Multi-view analysis** — combine multiple leaf images for robust diagnosis
- **Environmental data fusion** — integrate temperature, humidity, soil data
- **Hyperspectral imaging** support for early disease detection

### Deployment & Scaling
- **Docker Compose** for containerized deployment
- **Kubernetes** for horizontal scaling
- **CDN-backed model serving** for global deployment
- **Offline-capable PWA** (Progressive Web App) for areas with limited connectivity

### AI/ML Improvements
- **Vision Transformers** (ViT) as alternative to CNN
- **Self-supervised pretraining** on unlabeled agricultural images
- **Continual learning** — model improves with incoming data without catastrophic forgetting
- **Explainability** — Grad-CAM / SHAP for visual explanation of predictions
