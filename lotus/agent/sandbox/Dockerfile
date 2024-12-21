FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install lotus-ai fastapi uvicorn

# Create a non-root user
RUN useradd -m -u 1000 sandbox
USER sandbox

COPY server.py /app/server.py

CMD ["python", "/app/server.py"]
