FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_BASE_URL_PATH=solitaire-solver

RUN useradd --create-home --shell /bin/sh appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser main_app.py ./
COPY --chown=appuser:appuser solvers/ ./solvers/
COPY --chown=appuser:appuser ui/ ./ui/

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os, urllib.request; p = os.environ.get('STREAMLIT_SERVER_BASE_URL_PATH', '').strip('/'); url = f'http://localhost:8501/{p}/_stcore/health' if p else 'http://localhost:8501/_stcore/health'; urllib.request.urlopen(url, timeout=3)" || exit 1

CMD ["streamlit", "run", "main_app.py"]
