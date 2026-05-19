FROM python:3.11-slim

WORKDIR /app

# Install uv — much faster than pip
RUN pip install --no-cache-dir uv

# --- Heavy deps (fastembed + onnxruntime + numpy) in their own layer.
# This layer is cached and skipped on re-deploys as long as requirements-heavy.txt is unchanged.
COPY requirements-heavy.txt .
RUN uv pip install --system --no-cache -r requirements-heavy.txt

# Pre-download ONNX embedding model into a known path baked into the image
ENV FASTEMBED_CACHE_PATH=/app/.fastembed_cache
ENV PYTHONUNBUFFERED=1
RUN python -c "from fastembed import TextEmbedding; list(TextEmbedding('sentence-transformers/all-MiniLM-L6-v2').embed(['warmup']))"

# --- Light deps (fastapi, uvicorn, groq, etc.)
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Copy app source
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
