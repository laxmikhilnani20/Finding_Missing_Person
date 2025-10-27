# Use official Python runtime as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and video processing
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Update pip and install dependencies with retries and timeout settings
RUN python -m pip install --upgrade pip && \
    pip install --default-timeout=100 --retries=3 --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/missing_persons data/detections data/models config

# Expose application port
EXPOSE 8501

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;udp|timeout;20000000"
ENV OPENCV_LOG_LEVEL=DEBUG
ENV OMP_NUM_THREADS=4
ENV OPENBLAS_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

# Run the Flask application with gunicorn (production-ready config)
# Using gthread worker with 4 threads for handling multiple camera streams
CMD ["gunicorn", \
     "--workers", "1", \
     "--worker-class", "gthread", \
     "--threads", "4", \
     "--worker-tmp-dir", "/dev/shm", \
     "--bind", "0.0.0.0:8501", \
     "--timeout", "120", \
     "--keep-alive", "60", \
     "--log-level", "info", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "flask_app:app"]
