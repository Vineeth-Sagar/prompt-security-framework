FROM python:3.11-slim

WORKDIR /app

# tesseract-ocr: image handler OCR (Phase 1). ffmpeg: audio decoding for
# faster-whisper (Phase 1). Installed now so requirements.txt's pinned
# pytesseract/faster-whisper are usable as soon as those handlers land,
# without a second Dockerfile edit.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
