#!/usr/bin/env python3
"""Download ONNX model and inline external weights."""

from huggingface_hub import hf_hub_download
import onnx
from pathlib import Path

cache = '/root/.cache/scurl/models'

# Download model files
print('Downloading model files...')
model_path = hf_hub_download(
    'onnx-community/embeddinggemma-300m-ONNX',
    subfolder='onnx',
    filename='model.onnx',
    cache_dir=cache,
)
hf_hub_download(
    'onnx-community/embeddinggemma-300m-ONNX',
    subfolder='onnx',
    filename='model.onnx_data',
    cache_dir=cache,
)
hf_hub_download(
    'onnx-community/embeddinggemma-300m-ONNX',
    filename='tokenizer.json',
    cache_dir=cache,
)

# Load model with external data
print('Loading model with external data...')
model = onnx.load(model_path, load_external_data=True)

# Convert tensors to inline format
print('Converting tensors to inline format...')
for tensor in model.graph.initializer:
    if tensor.data_location == onnx.TensorProto.EXTERNAL:
        tensor.data_location = onnx.TensorProto.DEFAULT
        tensor.ClearField('external_data')

# Save model with inlined weights
print('Saving model with inlined weights...')
inlined_path = Path(model_path).parent / 'model_inlined.onnx'
onnx.save(model, str(inlined_path))

# Replace original with inlined version
Path(model_path).unlink()
Path(str(model_path) + '_data').unlink()
inlined_path.rename(model_path)

print('Done')
