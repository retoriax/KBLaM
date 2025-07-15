FROM python:3.10-slim

WORKDIR /app

COPY gen_wikipedia_data.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY entrypoint.sh .
COPY gen_wikipedia_data.py .

RUN chmod +x ./entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]