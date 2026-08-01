# 🎙️ Advanced AI Voice Studio

An AI-powered voice generation platform built with **Python**, **Gradio**, **Docker**, and **F5-TTS**. This project provides high-quality voice cloning, multi-speaker podcast generation, multilingual speech synthesis, audio editing, and machine learning audio preprocessing through an intuitive web interface.

---

## ✨ Features

### 🎤 Voice Cloning
- Clone voices using F5-TTS.
- Automatic reference audio processing.
- Reference transcript support.
- High-quality speech generation.

### 🎙️ Multi-Voice Podcast Generator
- Generate podcasts using multiple saved voices.
- Automatic speaker detection.
- Unknown speaker handling.
- Smooth audio stitching with configurable pauses.
- Validation for malformed scripts.

### 🌍 Hindi / Urdu Speech
- Supports Hindi and Urdu text generation.
- Automatic language detection.
- Microsoft Neural voice fallback when an RVC model is unavailable.

### ✂️ Audio Editor
- Trim audio.
- Cut selected regions.
- Replace sections with newly generated speech.
- Export edited audio.

### 🧠 Voice Training Studio
Machine learning preprocessing pipeline including:
- Audio normalization
- Noise reduction
- Silence removal
- Resampling to 16 kHz mono
- Automatic dataset chunking
- Training dataset preparation

---

# 🛠️ Technologies Used

- Python
- Gradio
- Docker
- F5-TTS
- Edge-TTS
- PyDub
- SoundFile
- FFmpeg
- Transformers
- Torch

---

# 📂 Project Structure

```
AI Voice Studio
│
├── app.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
├── inference_config.toml
├── rvc_infer.py
├── transliterate.py
├── demo.html
├── assets/
├── training_data/
├── saved_voices/
└── rvc_models/
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/RisalKalwar/advanced-ai-voice-studio.git
cd advanced-ai-voice-studio
```

Build Docker

```bash
docker compose up --build
```

Open the application

```
http://localhost:7860
```

---

# 📋 Usage

## Voice Cloner
1. Upload reference audio.
2. Enter the reference transcript.
3. Enter the text to generate.
4. Generate cloned speech.

---

## Multi-Voice Podcast

Example:

```
ARIA: Hello everyone.
REAL: Welcome to our AI Voice Studio.
ARIA: Thank you for listening.
```

The application automatically:
- Matches speakers with saved voices
- Generates each line
- Combines all audio into one podcast

---

## Voice Training Studio

Upload a recording to:

- Normalize volume
- Reduce noise
- Remove silence
- Resample to 16 kHz mono
- Split audio into training chunks

---

# 📷 Demo

This project demonstrates:

- Voice cloning
- Multi-speaker podcast generation
- Hindi/Urdu speech generation
- Audio editing
- Machine learning audio preprocessing

---

# 🔮 Future Improvements

- GPU acceleration
- Real-time voice conversion
- Additional multilingual voice models
- Emotion-aware speech synthesis
- Speaker diarization
- Advanced audio restoration

---

# 👨‍💻 Author

**Risal Kalwar**

GitHub:
https://github.com/RisalKalwar

---

# 📄 License

This project was developed for educational and research purposes.
