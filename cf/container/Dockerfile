FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install scurl from local wheel with browser extra
COPY *.whl /tmp/
RUN pip install --no-cache-dir "/tmp/sibylline_scurl-0.2.0-py3-none-any.whl[browser,prompt-defender]" numpy && \
    playwright install chromium --only-shell && \
    playwright install-deps chromium && \
    rm -f /tmp/*.whl

# Pre-download MiniLM model (86MB, no external data file)
COPY inline_model.py /tmp/
RUN python /tmp/inline_model.py && rm /tmp/inline_model.py

COPY server.py .

EXPOSE 8080

CMD ["python", "server.py"]
