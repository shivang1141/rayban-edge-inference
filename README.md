# Meta Ray-Ban Edge Inference Server

A high-performance, asynchronous edge-inference architecture designed to process data streams from Meta Ray-Ban Smart Glasses. It leverages a local PyTorch backend (with flexibility for Apple Silicon optimization) to execute Speech-to-Text (STT), Vision-Language Models (VLM), Large Language Models (LLM), and Text-to-Speech (TTS) with ultra-low latency.

## 🚀 Architecture Overview

This project is built to handle streaming audio and image data from the glasses, intelligently route requests to preserve compute resources, and stream responses back conversationally.

### The 4-Phase Pipeline

1. **Ingestion Pipeline (`app/main.py` & `app/services/stt.py`)**: 
   - An asynchronous FastAPI server holding open WebSocket connections.
   - Receives `base64` encoded audio payloads and transcribes them near-instantly using an 8-bit quantized `faster-whisper` model.
2. **Intelligent Routing (`app/services/router.py`)**: 
   - A lightweight NLP heuristic evaluates the incoming transcription.
   - If spatial/visual keywords (e.g., *"what is this?"*, *"look at that"*) are detected, it routes the payload to the heavier Vision model (`VISION_REQUIRED`).
   - Otherwise, it falls back to the faster, text-only Language model (`TEXT_ONLY`).
3. **Model Inference (`app/services/llm.py` & `app/services/vlm.py`)**:
   - Evaluates the query using PyTorch `transformers`. 
   - VLM (e.g., LLaVA) processes the image array and text context.
   - LLM (e.g., Llama 3 8B or Qwen) processes standard logic and conversational queries.
   - *(Note: Currently configured to initialize in `mock=True` mode for rapid testing without downloading large AI weights).*
4. **Audio Delivery (`app/services/tts.py`)**:
   - Converts the textual output from the models into natural-sounding speech.
   - Packages the output as a `base64` audio payload and sends it back to the glasses via WebSocket.

## 🛠 Prerequisites

- Python 3.10+
- A compatible OS (Windows/Mac/Linux)
- [Optional] An NVIDIA GPU (CUDA) or Apple Silicon Mac for real-time model inference.

## 📦 Installation & Setup

1. **Create and activate a virtual environment**:
   ```bash
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # On Mac/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃‍♂️ Running the Server

Start the FastAPI application using `uvicorn`:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
The server will now accept WebSocket connections at `ws://localhost:8000/ws/stream`.

## 🧪 Testing the Pipeline

You don't need the physical smart glasses to verify the backend flow. A `test_client.py` script is included to simulate payload delivery.

While the server is running, open a **new terminal** and execute:
```bash
python test_client.py
```

**Expected Output**: The test client will connect, push a dummy audio/image payload, and print out the server's processed response, including the simulated model output and detailed latency metrics.

## ⚙️ Configuration & "Mock" Mode

By default, the heavy AI models (LLM, VLM, and TTS) are initialized in "mock" mode via `app/config.py`. This allows the server to boot instantly and handle requests without allocating massive amounts of VRAM or downloading Gigabytes of weights from Hugging Face.

To run real local inference:
1. Copy the `.env.example` file to a new file named `.env`:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and change the mock settings to `False`:
   ```env
   MOCK_LLM=False
   MOCK_VLM=False
   MOCK_TTS=False
   ```
3. Restart the server. *(Note: The initial start will take significantly longer as it downloads the default models to your PyTorch cache).*

## 🔮 Future Roadmap

- **Apple Silicon Optimization**: Transition PyTorch `transformers` logic over to Apple's `mlx-lm` and `mlx-vlm` to unlock massive unified memory performance gains on Mac hardware.
- **Chunked Audio Streaming**: Refactor the TTS pipeline to stream audio byte-chunks as soon as token sentences are generated, creating zero-perceived latency for the user.
- **Meta SDK Integration**: Wire the WebSocket directly into the Meta SDK event webhooks once access is fully provisioned.
