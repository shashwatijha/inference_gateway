# Build from the repo root: docker build -f worker/Dockerfile .
FROM python:3.12-slim
WORKDIR /srv
COPY worker/requirements.txt worker/requirements.txt
RUN pip install --no-cache-dir -r worker/requirements.txt
COPY worker worker
RUN useradd -r app && chown -R app /srv
USER app
EXPOSE 9100
CMD ["python", "-m", "worker.main"]
