# alertcal API image. Build: docker build -t alertcal:local .   Run: docker run --rm -p 8000:8000 alertcal:local
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Only what the API needs: package metadata + source. Data, tests and experiments stay out.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[api]"

RUN useradd --create-home --shell /usr/sbin/nologin app
USER app

EXPOSE 8000
CMD ["uvicorn", "alertcal.api:app", "--host", "0.0.0.0", "--port", "8000"]
