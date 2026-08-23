FROM python:3.11-slim

WORKDIR /app

# Install system build dependencies if necessary
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Run training to build initial baseline artifacts
RUN python training/train.py

EXPOSE 5000

ENV HOST=0.0.0.0
ENV PORT=5000
ENV REDIS_URL=redis://redis-service:6379

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]
