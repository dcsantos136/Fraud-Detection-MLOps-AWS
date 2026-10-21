FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 3000

CMD ["python", "-m", "bentoml", "serve", "bentoml_service.py:FraudDetectionService", "--host", "0.0.0.0", "--port", "3000"]
