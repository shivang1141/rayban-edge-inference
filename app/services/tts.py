import logging
import base64
import wave
import io

logger = logging.getLogger(__name__)

class TTSService:
    def __init__(self, mock=True):
        self.mock = mock
        if self.mock:
            logger.info("Initializing Mock TTS Service.")
        else:
            logger.info("Initializing TTS Engine (e.g., Piper/Kokoro)")
            # Initialize TTS engine here

    async def generate_audio_b64(self, text: str) -> str:
        """
        Converts text to speech and returns a base64 encoded audio chunk.
        """
        if self.mock:
            # Generate a tiny dummy WAV file in memory
            wav_io = io.BytesIO()
            with wave.open(wav_io, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                # Write 100 frames of silence
                wav_file.writeframes(b'\x00' * 200)
            
            wav_bytes = wav_io.getvalue()
            return base64.b64encode(wav_bytes).decode('utf-8')
        
        # Real TTS Generation logic here
        pass
