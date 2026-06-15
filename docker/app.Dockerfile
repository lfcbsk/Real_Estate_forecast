FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements.txt requirements.txt
RUN pip install --upgrade pip setuptools==69.5.1 wheel \
    && pip install --prefer-binary -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "src/app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
