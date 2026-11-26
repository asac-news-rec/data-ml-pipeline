# 추론 서버 이미지. 빌드 컨텍스트는 저장소 루트다.
FROM python:3.12-slim-bookworm

# LightGBM이 쓰는 OpenMP 런타임
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8.4 /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra serving && rm -rf /root/.cache/uv

COPY serving/ /app/serving/
COPY features/feast/feature_repo/ /app/features/feast/feature_repo/

RUN useradd --uid 10001 --create-home app
USER 10001
EXPOSE 8080
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8080"]
