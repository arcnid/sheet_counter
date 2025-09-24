# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HEADLESS=1

WORKDIR /app

# (Optional but recommended) OS deps that some wheels expect
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip and ensure we always install to THIS interpreter
RUN python -m pip install --upgrade pip

# Install PyTorch CPU wheels explicitly, then Ultralytics
RUN python -m pip install --no-cache-dir \
      torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cpu \
 && python -m pip install --no-cache-dir ultralytics==8.2.103 supabase==2.6.0 numpy \
 && python -m pip uninstall -y opencv-python || true \
 && python -m pip install --no-cache-dir opencv-python-headless==4.10.0.84

# Copy code AFTER deps for better caching
COPY . /app

# Start your script (make sure this file exists in repo)
CMD ["python", "count_container.py"]
