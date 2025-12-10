# Lightweight base image
FROM python:3.10-slim

WORKDIR /app

# Install system deps (optional, depends on your project)
RUN apt-get update && apt-get install -y \
    git wget build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy CUDA + PyTorch wheels
COPY whls/ /app/whls/

# Install wheels first (contains torch, torchvision, torchaudio, CUDA runtime, cuDNN, etc.)
RUN pip install --no-cache-dir /app/whls/*.whl

# Copy and install python deps
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . /app

# Create checkpoints dir
RUN mkdir -p /app/checkpoints /app/samples

# Streamlit port
EXPOSE 8501

# Streamlit settings
ENV STREAMLIT_SERVER_ENABLECORS=false
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501

# Run app
CMD ["streamlit", "run", "app_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
