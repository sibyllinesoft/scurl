FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install scurl from local wheel with browser extra
COPY *.whl /tmp/
RUN pip install --no-cache-dir "/tmp/sibylline_scurl-0.1.1-py3-none-any.whl[browser,prompt-defender]" numpy && \
    playwright install chromium --only-shell && \
    playwright install-deps chromium && \
    rm -f /tmp/*.whl

# Download ONNX model and convert to single file with inlined weights
# This avoids external data path issues at runtime
COPY inline_model.py /tmp/
RUN pip install --no-cache-dir onnx && \
    python /tmp/inline_model.py && \
    rm /tmp/inline_model.py && \
    pip uninstall -y onnx protobuf

COPY server.py .

EXPOSE 8080

CMD ["python", "server.py"]
