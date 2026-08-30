from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import logging
import time

from app.config import settings
from app.services.stt import STTService
from app.services.router import IntentRouter
from app.services.llm import LLMService
from app.services.vlm import VLMService
from app.services.tts import TTSService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Meta Ray-Ban Smart Glasses Backend", version="1.0.0")

# Initialize services
logger.info("Initializing services...")
# In a real environment, you might not mock these. 
# They are initialized based on the configuration (defaults to mocked).
stt_service = STTService(model_size=settings.stt_model_size)
llm_service = LLMService(model_id=settings.llm_model_id, mock=settings.mock_llm)
vlm_service = VLMService(model_id=settings.vlm_model_id, mock=settings.mock_vlm)
tts_service = TTSService(mock=settings.mock_tts)

@app.get("/")
async def root():
    return {"message": "Meta Ray-Ban Edge Inference Server is running."}

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established.")
    try:
        while True:
            # Expecting {"audio": "base64...", "image": "base64..." (optional)}
            data = await websocket.receive_json()
            logger.info(f"Received payload keys: {list(data.keys())}")
            
            if "audio" in data:
                audio_b64 = data["audio"]
                
                # --- PHASE 1: STT ---
                start_time = time.time()
                transcription = await stt_service.transcribe_base64(audio_b64)
                stt_latency = time.time() - start_time
                logger.info(f"Transcription: '{transcription}' (Latency: {stt_latency:.2f}s)")
                
                if not transcription:
                    continue

                # --- PHASE 2: Routing ---
                route = IntentRouter.classify_intent(transcription)
                
                # --- PHASE 3: Inference ---
                inference_start = time.time()
                response_text = ""
                
                if route == "VISION_REQUIRED":
                    if "image" in data and data["image"]:
                        response_text = await vlm_service.generate(transcription, data["image"])
                    else:
                        response_text = "I need an image to answer that, but none was provided."
                else:
                    response_text = await llm_service.generate(transcription)
                
                inference_latency = time.time() - inference_start
                logger.info(f"Model Response: '{response_text}' (Latency: {inference_latency:.2f}s)")
                
                # --- PHASE 4: TTS & Delivery ---
                tts_start = time.time()
                # For chunked TTS, we would yield audio chunks. Here we just generate the whole response.
                audio_response_b64 = await tts_service.generate_audio_b64(response_text)
                tts_latency = time.time() - tts_start
                logger.info(f"TTS Generated (Latency: {tts_latency:.2f}s)")
                
                # Send the final payload back to the glasses
                await websocket.send_json({
                    "status": "success",
                    "route": route,
                    "transcription": transcription,
                    "text_response": response_text,
                    "audio_response": audio_response_b64,
                    "metrics": {
                        "stt_latency": stt_latency,
                        "inference_latency": inference_latency,
                        "tts_latency": tts_latency,
                        "total_latency": time.time() - start_time
                    }
                })
            else:
                await websocket.send_json({"status": "error", "message": "No audio payload found."})

    except WebSocketDisconnect:
        logger.info("WebSocket connection closed.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
