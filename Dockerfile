# Dockerfile (GPU)
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

WORKDIR /app

COPY whls/ /app/whls/
RUN pip install /app/whls/*.whl

# copy files
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# Create checkpoints dir
RUN mkdir -p /app/checkpoints /app/samples

# Expose streamlit port
EXPOSE 8501

# Streamlit config to run in server mode
ENV STREAMLIT_SERVER_ENABLECORS=false
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501

CMD ["streamlit", "run", "app_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
