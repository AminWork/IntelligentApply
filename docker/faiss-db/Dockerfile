# ------------------ faiss-db ------------------
    FROM docker.arvancloud.ir/python:3.11-slim

    # native build tools for faiss-cpu wheel
    RUN apt-get update && \
        apt-get install -y --no-install-recommends build-essential && \
        rm -rf /var/lib/apt/lists/*
    
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt
    
    COPY app ./app
    ENV PYTHONUNBUFFERED=1
    CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
    