# Educational API image — not for production trading.
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY rl/requirements.txt /app/rl/requirements.txt
COPY requirements-api.txt /app/requirements-api.txt

# RL stack is heavy; install API deps first, then RL.
# torch CPU comes from rl/requirements.txt (pytorch.org/whl/cpu). PyPI torch is CUDA.
RUN pip install --upgrade pip \
    && pip install -r /app/requirements-api.txt \
    && pip install -r /app/rl/requirements.txt

COPY api /app/api
COPY rl /app/rl
COPY rag /app/rag
COPY streamlit_app /app/streamlit_app

EXPOSE 8000 8501

# Default: API. Override in compose for Streamlit.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
