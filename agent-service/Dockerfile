FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY agent-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared environment library
COPY environment /app/environment

# Copy agent service source
COPY agent-service/. .

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
