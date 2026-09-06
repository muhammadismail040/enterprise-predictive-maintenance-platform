FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATABASE_URL=sqlite:///./local.db \
    MLFLOW_TRACKING_URI=sqlite:///./mlflow.db

COPY frontend/package.json frontend/package-lock.json* ./frontend/
RUN apt-get update \
    && apt-get install -y --no-install-recommends nodejs npm \
    && cd frontend \
    && npm install

COPY frontend ./frontend
RUN cd frontend && npm run build

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "main.py"]
