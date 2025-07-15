#!/bin/sh

echo "WIKIDATA_API_KEY=$WIKIDATA_API_KEY" >> /app/.env
echo "WIKIDATA_PATH=$WIKIDATA_PATH" >> /app/.env
echo "WIKIDATA_MODEL_NAME=$WIKIDATA_MODEL_NAME" >> /app/.env

exec python gen_wikipedia_data.py "$@"