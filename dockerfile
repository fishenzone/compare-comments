FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY ./app /app/app

RUN mkdir -p /app/results /app/uploads && \
    chmod -R 777 /app/results /app/uploads
    
EXPOSE 8001
