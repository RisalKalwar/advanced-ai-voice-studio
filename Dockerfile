FROM python:3.11-slim

WORKDIR /app

# Install system packages
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker caching
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Install the requirements that work
RUN pip install --no-cache-dir \
    gradio \
    yt-dlp

# Install the remaining libraries used by the application
RUN pip install --no-cache-dir \
    edge-tts \
    pydub \
    soundfile \
    transliterate \
    transformers \
    torch \
    tomli_w \
    f5-tts \
    watchfiles \
    noisereduce

# Copy the project files
COPY . .

EXPOSE 7860

CMD ["python", "app.py"]