FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV GUI (libgl1, libglib2.0-0) + numpy (libgomp1)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Ensure CPU-only torch (ultralytics may pull CUDA torch by default)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps

COPY . .

EXPOSE 5000

ENV CONFIG=cameras.yaml
ENV PORT=5000

CMD ["sh", "-c", "python server.py --config \"$CONFIG\" --port \"$PORT\""]
