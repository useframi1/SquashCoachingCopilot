# Multi-stage build for squash video analysis API
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy and install local packages
COPY packages/*.whl /tmp/packages/
RUN pip install --no-cache-dir /tmp/packages/*.whl && rm -rf /tmp/packages

# Copy application code
COPY . .

# Create volumes directories
RUN mkdir -p /app/volumes/videos /app/volumes/output

# Expose port for API
EXPOSE 8000

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "api_main:app", "--host", "0.0.0.0", "--port", "8000"]
