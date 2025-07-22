FROM python:3.10-slim

WORKDIR /app

RUN echo "datasets"           > requirements.txt && \
    echo "langchain-core"     >> requirements.txt && \
    echo "tqdm"               >> requirements.txt && \
    echo "langchain-openai"   >> requirements.txt && \
    echo "python-dotenv"      >> requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY entrypoint.sh .
COPY gen_wikipedia_data.py .
COPY wikipedia_data_prompts.py .

RUN chmod +x ./entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]