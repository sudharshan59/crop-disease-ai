# System Architecture

## Overview

The **Crop Disease Detection & Smart Treatment Recommendation System** follows a modular, layered architecture designed for scalability, maintainability, and edge deployment readiness.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                              │
│                   Next.js 14 (App Router)                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐    │
│  │  Upload   │  │  Result  │  │  Loader  │  │  History Page │    │
│  │Component  │  │Component │  │Component │  │              │    │
│  └────┬─────┘  └────▲─────┘  └──────────┘  └──────┬───────┘    │
│       │              │                              │            │
│       ▼              │                              ▼            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │               Axios HTTP Client (lib/api.ts)             │    │
│  └──────────────────────┬──────────────────────────────────┘    │
└──────────────────────────┼──────────────────────────────────────┘
                           │  HTTP (JSON)
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                     FASTAPI BACKEND API                           │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Endpoints:                                                 │  │
│  │    GET  /api/health   → System health check                 │  │
│  │    POST /api/predict  → Disease prediction pipeline         │  │
│  │    GET  /api/history  → Prediction history (paginated)      │  │
│  └────────┬───────────────────────────────────────────────────┘  │
│           │                                                       │
│           ▼                                                       │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              PREDICTION PIPELINE                              │ │
│  │                                                               │ │
│  │  ┌─────────────┐    ┌──────────────┐    ┌───────────────┐   │ │
│  │  │   OpenCV     │    │  CNN Model   │    │   Severity    │   │ │
│  │  │Preprocessing │───▶│  (ResNet18)  │───▶│  Estimation   │   │ │
│  │  │             │    │              │    │               │   │ │
│  │  │ • Resize    │    │ • 16 classes │    │ • Confidence  │   │ │
│  │  │ • Blur      │    │ • Softmax    │    │ • Area ratio  │   │ │
│  │  │ • Normalize │    │ • Confidence │    │ • Combined    │   │ │
│  │  │ • Segment   │    │              │    │   scoring     │   │ │
│  │  └─────────────┘    └──────────────┘    └───────┬───────┘   │ │
│  │                                                  │           │ │
│  │                                                  ▼           │ │
│  │                                         ┌───────────────┐   │ │
│  │                                         │  LLM Engine   │   │ │
│  │                                         │               │   │ │
│  │                                         │ • GGUF (CPU)  │   │ │
│  │                                         │ • BnB (GPU)   │   │ │
│  │                                         │ • Fallback    │   │ │
│  │                                         └───────┬───────┘   │ │
│  │                                                  │           │ │
│  └──────────────────────────────────────────────────┼───────────┘ │
│                                                      │            │
│  ┌───────────────────────────────────────────────────▼──────────┐ │
│  │                    SQLite Database                            │ │
│  │  predictions(id, timestamp, filename, disease, confidence,   │ │
│  │              severity, recommendation_json, region)          │ │
│  └──────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Prediction Request Flow

1. **Image Upload** → User uploads RGB leaf image via drag-and-drop or file picker
2. **API Receipt** → FastAPI receives `UploadFile` with optional `region` query parameter
3. **Preprocessing** → OpenCV pipeline:
   - Decode bytes → BGR numpy array
   - Resize to 224×224
   - Gaussian blur (5×5 kernel) for noise reduction
   - Optional green-channel leaf segmentation
   - BGR → RGB conversion, normalize to [0, 1]
   - ImageNet standardization (mean/std)
   - Convert to PyTorch tensor `[1, 3, 224, 224]`
4. **CNN Inference** → ResNet18 forward pass:
   - Produces logits for 16 classes
   - Softmax → probability distribution
   - Extract top-1 class and confidence score
5. **Severity Estimation** → Combined analysis:
   - CNN confidence score (40% weight)
   - Diseased area ratio from segmentation (60% weight)
   - Maps to: healthy / mild / moderate / severe
6. **Recommendation Generation** → LLM or fallback:
   - Build prompt with disease, confidence, severity, region
   - Generate structured JSON (description, causes, treatment, prevention)
   - Parse and validate output
   - Fallback to expert dictionary if LLM unavailable
7. **Database Storage** → Async SQLite write with all prediction data
8. **Response** → JSON with disease, confidence, severity, recommendation

### Concurrency Model

- FastAPI runs async endpoints with Uvicorn
- Blocking model inference is offloaded to ThreadPoolExecutor via `run_in_executor`
- Database operations use `aiosqlite` for non-blocking async access
- Models are loaded once at startup (lifespan context manager) and reused

## Component Responsibilities

| Component | File | Role |
|-----------|------|------|
| Config | `app/config.py` | Centralized settings, constants, env loading |
| Preprocessing | `app/preprocessing.py` | OpenCV image pipeline |
| CNN Model | `app/model.py` | ResNet18 loading and inference |
| Severity | `app/severity.py` | Disease severity estimation |
| LLM Engine | `app/llm.py` | Recommendation generation (multi-backend) |
| Database | `app/database.py` | Async SQLite operations |
| Schemas | `app/schemas.py` | Pydantic request/response models |
| Main | `app/main.py` | FastAPI app, routes, lifespan |
| Runner | `run.py` | Uvicorn startup entry point |
| Training | `train.py` | Full CNN training pipeline |
