FROM python:3.11-slim

WORKDIR /app

# System deps for later phases (pytesseract needs the tesseract binary,
# faster-whisper/audio needs libsndfile) are added when those handlers
# land, to keep this image lean until then.

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
