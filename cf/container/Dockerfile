FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install scurl from local wheel with browser extra
COPY *.whl /tmp/
RUN pip install --no-cache-dir "/tmp/sibylline_scurl-0.1.1-py3-none-any.whl[browser,prompt-defender]" numpy && \
    playwright install chromium --only-shell && \
    playwright install-deps chromium && \
    rm -f /tmp/*.whl

# Note: ONNX model (~1.2GB) downloads on first request to stay under 2GB image limit
# Cold start will be slower but subsequent requests use cached model

COPY server.py .

EXPOSE 8080

CMD ["python", "server.py"]
