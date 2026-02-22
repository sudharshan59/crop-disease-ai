# 🌿 AI-Based Crop Disease Detection & Smart Treatment Recommendation System

> **IEEE Paper Implementation** — End-to-end deep learning pipeline for automated crop disease diagnosis with LLM-powered treatment recommendations.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                         FRONTEND                                 │
│     Next.js 14  ·  TypeScript  ·  TailwindCSS                    │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐  ┌──────────────┐   │
│  │  Upload   │→│  Loader  │→│   Result    │  │   History     │   │
│  │ (Dropzone)│  │(Spinner) │  │(Tabs+Badge)│  │ (Pagination)  │   │
│  └──────────┘  └──────────┘  └────────────┘  └──────────────┘   │
└──────────────────────┬───────────────────────────────────────────┘
                       │  HTTP / REST
┌──────────────────────▼───────────────────────────────────────────┐
│                         BACKEND                                  │
│     FastAPI  ·  PyTorch  ·  OpenCV  ·  llama-cpp-python          │
│                                                                  │
│  ┌────────────┐  ┌────────────┐  ┌───────────┐  ┌────────────┐  │
│  │ Preprocess │→│  ResNet18  │→│  Severity  │→│  LLM / Dict  │  │
│  │  (OpenCV)  │  │  (CNN)     │  │ Estimator │  │ (phi-2 GGUF)│  │
│  └────────────┘  └────────────┘  └───────────┘  └─────┬──────┘  │
│                                                        │         │
│  ┌─────────────────────────────────────────────────────▼──────┐  │
│  │                   SQLite (aiosqlite)                        │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Next.js 14, React 18, TypeScript, TailwindCSS, Axios, react-dropzone |
| **Backend** | FastAPI, Uvicorn, Pydantic v2 |
| **CNN Model** | PyTorch ResNet18 (ImageNet pretrained), TorchScript export |
| **Preprocessing** | OpenCV (resize, blur, HSV segmentation, normalization) |
| **LLM** | phi-2 via llama-cpp-python (GGUF Q4\_K\_M) or bitsandbytes 4-bit |
| **Database** | SQLite via aiosqlite |
| **Disease Classes** | 16 (15 diseases + 1 Healthy) across 5 crops |

## Supported Diseases (16 Classes)

| # | Disease | Crop |
|---|---------|------|
| 1 | Apple Scab | Apple |
| 2 | Apple Black Rot | Apple |
| 3 | Apple Cedar Rust | Apple |
| 4 | Corn Cercospora Leaf Spot | Corn |
| 5 | Corn Common Rust | Corn |
| 6 | Corn Northern Leaf Blight | Corn |
| 7 | Grape Black Rot | Grape |
| 8 | Grape Esca (Black Measles) | Grape |
| 9 | Grape Leaf Blight | Grape |
| 10 | Potato Early Blight | Potato |
| 11 | Potato Late Blight | Potato |
| 12 | Tomato Bacterial Spot | Tomato |
| 13 | Tomato Early Blight | Tomato |
| 14 | Tomato Late Blight | Tomato |
| 15 | Tomato Leaf Mold | Tomato |
| 16 | Healthy | — |

---

## Quick Start

### Prerequisites

- Python 3.10+ ([Download](https://www.python.org/downloads/))
- Node.js 18+ ([Download](https://nodejs.org/))
- Git
- (Optional) NVIDIA GPU with CUDA for faster inference

---

### Backend Setup

#### Step 1: Navigate to backend

```bash
cd crop-disease-ai/backend
```

#### Step 2: Create and activate virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `llama-cpp-python` requires a prebuilt wheel on Windows. If it fails to build, install it separately:
> ```bash
> pip install llama-cpp-python --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
> ```

#### Step 4: Copy environment config

**Windows:**
```powershell
copy .env.example .env
```

**Linux / macOS:**
```bash
cp .env.example .env
```

#### Step 5: Download phi-2 LLM model (optional but recommended)

```bash
pip install huggingface-hub
```

**Windows (PowerShell):**
```powershell
cd models
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='TheBloke/phi-2-GGUF', filename='phi-2.Q4_K_M.gguf', local_dir='.')"
cd ..
```

**Linux / macOS:**
```bash
cd models
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='TheBloke/phi-2-GGUF', filename='phi-2.Q4_K_M.gguf', local_dir='.')"
cd ..
```

> The file is ~1.8 GB. Without it, the system uses a built-in expert fallback dictionary and **works fully without any LLM**.

#### Step 6: Train the CNN model (optional for demo)

```bash
# Download PlantVillage dataset and extract to a folder
# Expected structure: dataset_root/ClassName/image.jpg

python train.py --data-dir /path/to/plantvillage --epochs 25 --batch-size 32
```

This saves:
- `models/disease_model_scripted.pt` -- TorchScript model
- `models/disease_model_best.pth` -- Checkpoint
- `models/class_mapping.json` -- Class index mapping
- `models/training_history.json` -- Loss/accuracy curves

> **Without training**, the system runs in demo mode using an ImageNet-pretrained backbone. Predictions will vary per image but won't be accurate.

#### Step 7: Start the backend server

```bash
python run.py
```

You should see:
```
  Host:   0.0.0.0
  Port:   8000
  Device: cpu
  LLM:    gguf (or fallback)
```

- API server: http://localhost:8000
- Swagger docs: http://localhost:8000/docs
- Health check: http://localhost:8000/api/health

---

### Frontend Setup

Open a **new terminal** (keep the backend running).

#### Step 1: Navigate to frontend

```bash
cd crop-disease-ai/frontend
```

#### Step 2: Install Node.js dependencies

```bash
npm install
```

#### Step 3: Start development server

```bash
npm run dev
```

You should see:
```
  Local: http://localhost:3000
```

#### Step 4: Open in browser

- Home (Diagnose): http://localhost:3000
- History: http://localhost:3000/history

---

### Verify Everything Works

1. Open http://localhost:3000 in your browser
2. Drag and drop (or click to select) a leaf image (JPEG/PNG, max 10MB)
3. Click **"Analyze Disease"**
4. View disease prediction, confidence score, severity badge, and AI recommendations
5. Check http://localhost:3000/history for past predictions

### Common Issues

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'uvicorn'` | Activate venv first: `.\venv\Scripts\Activate.ps1` |
| `llama-cpp-python` build fails | Install prebuilt wheel (see Step 3 note above) |
| Port 8000 already in use | Kill existing process: `Get-NetTCPConnection -LocalPort 8000 \| % { Stop-Process -Id $_.OwningProcess -Force }` |
| LLM shows "fallback" | Download phi-2 model (Step 5) or set `LLM_BACKEND=none` in `.env` |
| Frontend can't reach backend | Ensure backend runs on port 8000 and check `frontend/.env.local` has `NEXT_PUBLIC_API_URL=http://localhost:8000` |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check — model & LLM status |
| `POST` | `/api/predict` | Upload image → disease prediction + recommendations |
| `GET` | `/api/history` | Paginated prediction history |

### POST /api/predict

**Request:** `multipart/form-data`
- `file` (required): JPEG/PNG image
- `region` (optional): Geographic region for localized advice

**Response:**
```json
{
  "disease": "Tomato Late Blight",
  "confidence": 0.943,
  "severity": "severe",
  "recommendation": {
    "description": "Late blight is caused by...",
    "causes": "Phytophthora infestans...",
    "treatment": "Apply copper-based fungicide...",
    "prevention": "Use resistant varieties..."
  },
  "timestamp": "2024-01-15T10:30:00",
  "filename": "leaf_sample.jpg"
}
```

---

## Project Structure

```
crop-disease-ai/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py          # Settings & disease classes
│   │   ├── preprocessing.py   # OpenCV image pipeline
│   │   ├── model.py           # ResNet18 classifier
│   │   ├── severity.py        # Severity estimation
│   │   ├── llm.py             # LLM engine + fallback dict
│   │   ├── database.py        # Async SQLite operations
│   │   ├── schemas.py         # Pydantic response models
│   │   └── main.py            # FastAPI app & endpoints
│   ├── context/
│   │   ├── SYSTEM_ARCHITECTURE.md
│   │   ├── MODEL_DETAILS.md
│   │   ├── FUTURE_WORK.md
│   │   └── DEPLOYMENT_GUIDE.md
│   ├── models/                # Model weights directory
│   ├── train.py               # Training script
│   ├── run.py                 # Uvicorn entry point
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── app/
│   │   ├── components/
│   │   │   ├── Upload.tsx     # Drag-and-drop upload
│   │   │   ├── Result.tsx     # Prediction results + tabs
│   │   │   ├── Loader.tsx     # Loading animation
│   │   │   └── SeverityBadge.tsx
│   │   ├── history/
│   │   │   └── page.tsx       # Prediction history
│   │   ├── layout.tsx         # Root layout
│   │   ├── page.tsx           # Home / diagnosis page
│   │   └── globals.css
│   ├── lib/
│   │   └── api.ts             # Axios API client & types
│   ├── package.json
│   ├── tailwind.config.ts
│   └── tsconfig.json
└── README.md
```

---

## LLM Configuration

The system supports two LLM backends (configurable via `LLM_BACKEND` in `.env`):

| Backend | Best For | Requirements |
|---------|----------|-------------|
| `gguf` (default) | CPU / Windows / Low RAM | llama-cpp-python |
| `bitsandbytes` | NVIDIA GPU | transformers, bitsandbytes, accelerate, CUDA |

Set `LLM_BACKEND=none` to disable the LLM entirely and rely on the built-in expert dictionary.

---

## Training Details

- **Architecture:** ResNet18 with custom fully-connected head (512 → 16)
- **Strategy:** Freeze backbone for 5 epochs → unfreeze all layers
- **Optimizer:** Adam (lr=1e-4, weight_decay=1e-5)
- **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=3)
- **Early Stopping:** Patience 5, monitors validation loss
- **Augmentation:** RandomResizedCrop, HorizontalFlip, Rotation(±15°), ColorJitter
- **Export:** TorchScript (`.pt`) for production inference

---

## Future Work

1. **Multi-Dataset Fusion** — Combine PlantVillage with FGVC, iNaturalist datasets
2. **Severity-Based Dosage** — Calibrate treatment quantities to severity level
3. **Edge Deployment** — ONNX / TFLite / CoreML export for mobile
4. **Regional Personalization** — Weather API + location-aware recommendations
5. **Prompt Logging & RAG** — Retrieval-augmented generation for evolving advice
6. **Modular Crop Expansion** — Plug-in architecture for new crops/diseases
7. **Farmer Feedback Loop** — Collect user corrections to improve predictions

See [`backend/context/FUTURE_WORK.md`](backend/context/FUTURE_WORK.md) for detailed plans.

---

## References

- **IEEE Paper:** "AI-Based Crop Disease Detection and Smart Treatment Recommendation System"
- **PlantVillage Dataset:** Hughes & Salathé (2015)
- **ResNet:** He et al., "Deep Residual Learning for Image Recognition" (2015)
- **phi-2:** Microsoft Research (2023)

---

## License

MIT License — See [LICENSE](LICENSE) for details.
