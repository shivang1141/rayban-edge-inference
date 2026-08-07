import asyncio
import websockets
import json
import base64
import wave
import io
import time

async def test_server():
    uri = "ws://localhost:8000/ws/stream"
    
    # 1. Create a dummy audio payload (1 second of silence, 16kHz, 16-bit mono)
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b'\x00' * 16000 * 2) # 1 second silence
    
    audio_b64 = base64.b64encode(wav_io.getvalue()).decode('utf-8')
    
    # 2. Create a dummy image payload (1x1 pixel)
    image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="

    async with websockets.connect(uri) as websocket:
        print("Connected to server.")
        
        payload = {
            "audio": audio_b64,
            "image": image_b64
        }
        
        print("Sending payload...")
        start_time = time.time()
        await websocket.send(json.dumps(payload))
        
        response = await websocket.recv()
        end_time = time.time()
        
        print(f"Received response in {end_time - start_time:.2f}s:")
        res_json = json.loads(response)
        
        # Truncate audio_response to avoid console spam
        if "audio_response" in res_json:
            res_json["audio_response"] = res_json["audio_response"][:20] + "..."
            
        print(json.dumps(res_json, indent=2))

if __name__ == "__main__":
    asyncio.run(test_server())
