# Deployment Guide

## Table of Contents

1. [Local Development Setup](#1-local-development-setup)
2. [Docker Deployment](#2-docker-deployment)
3. [GPU Deployment](#3-gpu-deployment)
4. [Production Deployment](#4-production-deployment)
5. [Environment Variables Reference](#5-environment-variables-reference)

---

## 1. Local Development Setup

### Prerequisites

- **Python:** 3.10+ (recommended 3.11)
- **Node.js:** 18+ (recommended 20 LTS)
- **npm:** 9+
- **Git:** 2.30+
- **RAM:** Minimum 4 GB (8 GB recommended for LLM features)
- **Disk:** ~3 GB for models and dependencies

### Backend Setup

```bash
# Clone the repository
git clone <repository-url>
cd crop-disease-ai

# Create Python virtual environment
cd backend
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment configuration
cp .env.example .env
# Edit .env as needed

# (Optional) Download LLM model for AI recommendations
# Visit: https://huggingface.co/TheBloke/phi-2-GGUF
# Download phi-2.Q4_K_M.gguf (~1.8 GB) into backend/models/

# (Optional) Train the CNN model
python train.py --data_dir /path/to/plantvillage --epochs 30

# Start the backend server
python run.py
```

The API will be available at `http://localhost:8000`.
API documentation at `http://localhost:8000/docs`.

### Frontend Setup

```bash
# In a new terminal
cd crop-disease-ai/frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will be available at `http://localhost:3000`.

### Quick Verification

1. Open `http://localhost:8000/api/health` — should return `{"status": "ok", ...}`
2. Open `http://localhost:3000` — should show the upload interface
3. Upload a leaf image — should receive disease prediction and recommendations

---

## 2. Docker Deployment

### Dockerfile (Backend)

```dockerfile
# backend/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "run.py"]
```

### Dockerfile (Frontend)

```dockerfile
# frontend/Dockerfile
FROM node:20-alpine

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm ci

COPY . .

RUN npm run build

EXPOSE 3000

CMD ["npm", "start"]
```

### Docker Compose

```yaml
# docker-compose.yml (project root)
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    volumes:
      - ./backend/models:/app/models
      - ./backend/data:/app/data
    environment:
      - API_HOST=0.0.0.0
      - API_PORT=8000
      - DEVICE=cpu
      - LLM_BACKEND=gguf
    restart: unless-stopped

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
    depends_on:
      - backend
    restart: unless-stopped
```

### Running with Docker

```bash
# Build and start all services
docker compose up --build

# Run in detached mode
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down
```

---

## 3. GPU Deployment

### NVIDIA GPU Setup

#### Prerequisites
- NVIDIA GPU with CUDA support (compute capability 3.5+)
- NVIDIA Driver 525+ installed
- NVIDIA Container Toolkit (for Docker)

#### Local GPU Setup

```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install bitsandbytes for GPU quantization
pip install bitsandbytes>=0.43.0

# Update .env for GPU mode
DEVICE=cuda
LLM_BACKEND=bitsandbytes
LLM_MODEL_NAME=microsoft/phi-2

# Start server
python run.py
```

#### Docker GPU Setup

```yaml
# docker-compose.gpu.yml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    volumes:
      - ./backend/models:/app/models
      - ./backend/data:/app/data
    environment:
      - DEVICE=cuda
      - LLM_BACKEND=bitsandbytes
      - LLM_MODEL_NAME=microsoft/phi-2
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
```

```bash
# Install NVIDIA Container Toolkit
# See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# Run with GPU
docker compose -f docker-compose.gpu.yml up --build
```

### Memory Requirements

| Configuration | CNN | LLM | Total RAM/VRAM |
|--------------|-----|-----|---------------|
| CPU (GGUF, no LLM) | ~200 MB | 0 MB | ~500 MB |
| CPU (GGUF, phi-2) | ~200 MB | ~2 GB | ~3 GB |
| GPU (BnB, phi-2) | ~200 MB | ~3 GB VRAM | ~4 GB VRAM |
| GPU (BnB, Mistral-7B) | ~200 MB | ~5 GB VRAM | ~6 GB VRAM |

---

## 4. Production Deployment

### Recommended Production Stack

```
Internet
   │
   ▼
┌──────────┐
│  Nginx   │ ← Reverse proxy, SSL termination, static files
└────┬─────┘
     │
     ├─── /api/* ──▶ Uvicorn (FastAPI backend, port 8000)
     │
     └─── /* ──────▶ Next.js (frontend, port 3000)
```

### Nginx Configuration

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Frontend
    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
    }

    # Backend API
    location /api/ {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        client_max_body_size 10M;
    }
}
```

### Production Checklist

- [ ] Set `API_RELOAD=False` in production
- [ ] Configure proper CORS origins (not wildcard)
- [ ] Enable HTTPS via SSL certificate
- [ ] Set `LOG_LEVEL=WARNING` for production
- [ ] Use process manager (systemd, PM2, or Supervisor)
- [ ] Set up health monitoring (check `/api/health` endpoint)
- [ ] Configure database backups for SQLite
- [ ] Set file upload size limits
- [ ] Enable rate limiting

---

## 5. Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | API server bind address |
| `API_PORT` | `8000` | API server port |
| `API_RELOAD` | `False` | Auto-reload on code changes (dev only) |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated allowed origins |
| `MODEL_PATH` | `models/plant_disease_model.pth` | CNN model weights path |
| `TORCHSCRIPT_PATH` | `models/plant_disease_model_scripted.pt` | TorchScript model path |
| `DEVICE` | `auto` | Compute device: `auto`, `cpu`, `cuda` |
| `LLM_BACKEND` | `gguf` | LLM backend: `gguf` or `bitsandbytes` |
| `LLM_MODEL_NAME` | `microsoft/phi-2` | HuggingFace model name (BnB mode) |
| `LLM_GGUF_PATH` | `models/phi-2.Q4_K_M.gguf` | GGUF model file path |
| `LLM_MAX_TOKENS` | `512` | Max LLM generation tokens |
| `LLM_TEMPERATURE` | `0.3` | LLM sampling temperature |
| `LLM_CONTEXT_LENGTH` | `2048` | GGUF context window size |
| `DB_PATH` | `data/predictions.db` | SQLite database file path |
| `LOG_LEVEL` | `INFO` | Logging level |
| `LOG_PROMPTS` | `True` | Enable LLM prompt logging |
