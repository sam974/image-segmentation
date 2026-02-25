# Utilisez une image plus récente pour la compatibilité FastAPI/Uvicorn
FROM python:3.11-slim-bookworm

WORKDIR /app

# Installation des dépendances système nécessaires à Pillow/OpenCV si besoin
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenjp2-7 libtiff5 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "segmentit_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]