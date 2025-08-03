FROM python:3.10-slim

USER root

RUN apt-get update && apt-get install -y gcc

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# ✅ 전체 프로젝트 코드 복사 (app 포함)
COPY . /app

WORKDIR /app

ENV PYTHONPATH=/app
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "7000"]
