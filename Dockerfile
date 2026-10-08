FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    OLLAMA_BASE_URL=http://host.docker.internal:11434/v1 \
    OLLAMA_MODEL=qwen3-vl:8b

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 7860

CMD ["python", "app.py"]
