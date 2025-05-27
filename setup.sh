#!/bin/bash
set -e

LOG_FILE="setup.log"
pip3 install --upgrade pip >> "$LOG_FILE" 2>&1

pip3 install --upgrade "huggingface-hub[cli]" sentence-transformers >> "$LOG_FILE" 2>&1
echo "[Setup] Package installation complete." | tee -a "$LOG_FILE"

LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
huggingface-cli download "$LLM_MODEL" --local-dir-use-symlinks False >> "$LOG_FILE" 2>&1
echo "[Setup] LLM download complete." | tee -a "$LOG_FILE"

EMBEDDING_MODEL="sergeyzh/BERTA"
python3 -c "
import sys
from sentence_transformers import SentenceTransformer

model_name = '${EMBEDDING_MODEL}'
try:
    SentenceTransformer(model_name)
except Exception as e:
    print(f'Error downloading/loading Embedding model {model_name}: {e}', file=sys.stderr)
    exit(1)
" >> "$LOG_FILE" 2>&1
echo "[Setup] Embedding Model download complete." | tee -a "$LOG_FILE"

mkdir -p ./qdrant_data
echo "[Setup] Directory ./qdrant_data created." | tee -a "$LOG_FILE"

echo "[Setup] Setup complete." | tee -a "$LOG_FILE"
