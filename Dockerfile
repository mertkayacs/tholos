FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11.17 /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON_DOWNLOADS=0
WORKDIR /build
COPY pyproject.toml uv.lock LICENSE ./
COPY tholos/ ./tholos/
RUN uv sync --locked --no-dev --no-editable --no-cache

FROM python:3.12-slim
RUN groupadd --gid 10001 tholos \
    && useradd --uid 10001 --gid tholos --no-log-init --no-create-home \
        --home-dir /data --shell /usr/sbin/nologin tholos \
    && mkdir /data \
    && chown tholos:tholos /data
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    THOLOS_HOME=/data \
    THOLOS_HOST=0.0.0.0
# Behind a TLS proxy uvicorn sees plain http, so set THOLOS_COOKIE_SECURE=1 to force the
# Secure flag on the session cookie.
WORKDIR /data
USER tholos
VOLUME ["/data"]
EXPOSE 7070
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; r = urllib.request.urlopen('http://127.0.0.1:7070/healthz', timeout=3); assert r.status == 200 and r.read() == b'ok'"]
CMD ["tholos"]
