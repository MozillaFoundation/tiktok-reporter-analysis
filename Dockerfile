FROM python:3.11.2-slim-buster

WORKDIR /app

# Install OpenCV dependencies
RUN apt-get update && apt-get install -y ffmpeg

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
