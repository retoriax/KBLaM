# Base image mit Python und pip
FROM python:3.12-slim

# Setze Arbeitsverzeichnis
WORKDIR /app
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Kopiere requirements und installiere sie
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Kopiere den Rest des Codes
COPY . .

# Standard-Entrypoint ist dein Skript
ENTRYPOINT ["python", "-u", "synthetic_data_chatopenai.py"]