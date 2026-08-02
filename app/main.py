from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Meta Ray-Ban Smart Glasses Backend", version="1.0.0")

@app.get("/")
async def root():
    return {"message": "Meta Ray-Ban Edge Inference Server is running."}

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established.")
    try:
        while True:
            data = await websocket.receive_json()
            logger.info(f"Received payload keys: {list(data.keys())}")
            await websocket.send_json({"status": "success", "received": list(data.keys())})

    except WebSocketDisconnect:
        logger.info("WebSocket connection closed.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
