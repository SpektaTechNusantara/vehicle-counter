FROM python:3.11-slim

WORKDIR /app

# Minimal system deps for OpenCV (headless/server), numpy, and onnxruntime
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Ensure CPU-only torch (ultralytics may pull CUDA torch by default)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps

# Replace opencv-python with headless (no X11/GL deps needed on server)
RUN pip install --no-cache-dir opencv-python-headless && \
    pip uninstall -y opencv-python

COPY . .

EXPOSE 5000

ENV CONFIG=cameras.yaml
ENV PORT=5000

CMD ["sh", "-c", "python server.py --config \"$CONFIG\" --port \"$PORT\""]
