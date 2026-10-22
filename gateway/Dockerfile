# Build from the repo root: docker build -f gateway/Dockerfile .
FROM python:3.12-slim
WORKDIR /srv
COPY gateway/requirements.txt gateway/requirements.txt
RUN pip install --no-cache-dir -r gateway/requirements.txt
COPY gateway gateway
RUN useradd -r app && chown -R app /srv
USER app
EXPOSE 8000
CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
