#!/bin/sh

echo "PROXY_API_KEY=$PROXY_API_KEY" > /app/.env
echo "PROXY_PATH=$PROXY_PATH" >> /app/.env

exec python gen_wikipedia_data.py "$@"