FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml .
COPY README.md .

RUN pip install --no-cache-dir -e .
RUN pip install wandb

COPY . .

RUN chmod +x ./entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]