FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml poetry.lock* requirements.txt* /app/

RUN pip install --upgrade pip \
    && if [ -f requirements.txt ]; then pip install -r requirements.txt; fi \
    && if [ -f poetry.lock ]; then pip install poetry && poetry export --without-hashes -f requirements.txt | pip install -r /dev/stdin; fi \
    && if [ -f pyproject.toml ] && [ ! -f poetry.lock ]; then pip install .; fi

COPY . /app

EXPOSE 8000

CMD ["python", "serve.py", "--config", "configs/deploy.yaml", "--checkpoint", "outputs/latest.ckpt"]
