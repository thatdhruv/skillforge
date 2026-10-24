FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml README.md* ./
COPY src ./src
RUN pip install .
RUN useradd -r -u 10001 skillforge
USER skillforge
ENTRYPOINT ["skillforge"]
