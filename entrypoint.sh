#!/bin/sh

# Login bei Weights & Biases
wandb login "$WANDB_API_KEY"

# Schreibe .env-Datei mit HF_TOKEN zur Laufzeit
echo "HF_TOKEN=$HF_TOKEN" > /app/.env

# Starte das Training mit allen durchgereichten Argumenten
exec python experiments/train.py "$@"