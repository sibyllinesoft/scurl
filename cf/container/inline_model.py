#!/usr/bin/env python3
"""Download MiniLM model for prompt injection detection."""

from huggingface_hub import hf_hub_download
from pathlib import Path

cache = '/root/.cache/scurl/models'

# Download MiniLM-L6-v2 (86MB, single file - no external data)
print('Downloading MiniLM-L6-v2 model...')
model_path = hf_hub_download(
    'Qdrant/all-MiniLM-L6-v2-onnx',
    filename='model.onnx',
    cache_dir=cache,
)
tokenizer_path = hf_hub_download(
    'Qdrant/all-MiniLM-L6-v2-onnx',
    filename='tokenizer.json',
    cache_dir=cache,
)

print(f'Model downloaded to: {model_path}')
print(f'Tokenizer downloaded to: {tokenizer_path}')
print('Done')
