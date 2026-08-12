from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import logging

from app.config import settings
from app.services.stt import STTService
from app.services.router import IntentRouter
from app.services.llm import LLMService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Meta Ray-Ban Smart Glasses Backend", version="1.0.0")

# Initialize services
logger.info("Initializing services...")
stt_service = STTService(model_size=settings.stt_model_size)
llm_service = LLMService(model_id=settings.llm_model_id, mock=settings.mock_llm)

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
                transcription = await stt_service.transcribe_base64(audio_b64)
                logger.info(f"Transcription: '{transcription}'")

                # --- PHASE 2: Routing ---
                route = IntentRouter.classify_intent(transcription)

                # --- PHASE 3: Inference ---
                response_text = await llm_service.generate(transcription)
                logger.info(f"Model Response: '{response_text}'")

                await websocket.send_json({
                    "status": "success",
                    "route": route,
                    "transcription": transcription,
                    "text_response": response_text
                })
            else:
                await websocket.send_json({"status": "error", "message": "No audio payload found."})

    except WebSocketDisconnect:
        logger.info("WebSocket connection closed.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
