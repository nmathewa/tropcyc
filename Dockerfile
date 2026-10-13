# Live forecast server. Train first (tropcyc train ... -o models/live.joblib), then:
#   docker build -t tropcyc . && docker run -p 8000:8000 -v $PWD/models:/app/models tropcyc
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[server]" && useradd --create-home --uid 10001 app
USER app
EXPOSE 8000
CMD ["tropcyc", "serve", "--model", "models/live.joblib", "--host", "0.0.0.0", "--port", "8000"]
