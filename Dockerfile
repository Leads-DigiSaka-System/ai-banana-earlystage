# Use Python 3.12 slim image as base (matches pyproject.toml)
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# Note: libgl1-mesa-glx replaced with libgl1 in newer Debian
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package installer)
RUN pip install uv

# Copy uv files first (for better caching)
COPY pyproject.toml uv.lock ./

# Install Python dependencies using uv (much faster than pip)
# Using --system flag to install system-wide (not in virtual env)
# uv automatically uses uv.lock if present
RUN uv pip install --system .

# Copy application code
COPY . .

# Create models directory if it doesn't exist
RUN mkdir -p models/weights

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

