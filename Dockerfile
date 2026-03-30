# Multi-stage Dockerfile for Crop Disease AI backend
# - Uses the lightweight Python slim image
# - Installs system libs required for OpenCV and basic utilities
# - Copies the `backend/` folder and installs Python dependencies

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install runtime dependencies required by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    git \
 && rm -rf /var/lib/apt/lists/*

# Copy backend sources
COPY backend/ /app/

# Upgrade pip and install Python dependencies
RUN python -m pip install --upgrade pip setuptools wheel
RUN if [ -f /app/requirements.txt ]; then pip install -r /app/requirements.txt; fi

# Expose the default API port used by the app (settings may override)
EXPOSE 8000

# Default working directory is /app; entrypoint runs the backend runner
CMD ["python", "run.py"]
