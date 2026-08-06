from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import logging

from app.config import settings
from app.services.stt import STTService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Meta Ray-Ban Smart Glasses Backend", version="1.0.0")

# Initialize services
logger.info("Initializing services...")
stt_service = STTService(model_size=settings.stt_model_size)

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
                transcription = await stt_service.transcribe_base64(data["audio"])
                logger.info(f"Transcription: '{transcription}'")

                await websocket.send_json({
                    "status": "success",
                    "transcription": transcription
                })
            else:
                await websocket.send_json({"status": "error", "message": "No audio payload found."})

    except WebSocketDisconnect:
        logger.info("WebSocket connection closed.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
