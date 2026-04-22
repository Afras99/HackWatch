FROM python:3.11-slim

WORKDIR /app

# System deps for GitPython and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[server]"

COPY . .

# Pre-build demo static assets if frontend source exists
RUN if [ -d "demo/build" ]; then echo "Demo build present"; else mkdir -p demo/build && echo "<html><body>HackWatch Demo</body></html>" > demo/build/index.html; fi

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
