# syntax=docker/dockerfile:1

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HEADLESS=1

WORKDIR /app

# Install base deps first so they cache
RUN pip install --upgrade pip

# Install deps; ultralytics may pull opencv-python (GUI) — remove it and ensure headless
RUN pip install --no-cache-dir ultralytics==8.2.103 supabase==2.6.0 numpy \
 && pip uninstall -y opencv-python || true \
 && pip install --no-cache-dir opencv-python-headless==4.10.0.84

# Copy your code & weights
COPY . /app

# Run your script
CMD ["python", "count_container.py"]
