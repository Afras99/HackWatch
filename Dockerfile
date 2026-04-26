FROM python:3.11-slim

WORKDIR /app

# System deps for GitPython
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer cache)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Install the hackwatch package (editable, no extras — deps already installed above)
RUN pip install --no-cache-dir -e "." --no-deps

# Ensure demo build dir exists
RUN mkdir -p demo/build

EXPOSE 8000

# HF Spaces runs as a non-root user; ensure writable dirs
RUN chmod -R 777 /app

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
