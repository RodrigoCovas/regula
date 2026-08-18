FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/src ./backend/src

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "backend.src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
