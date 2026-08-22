from faster_whisper import WhisperModel
import logging
import io
import base64

logger = logging.getLogger(__name__)

class STTService:
    def __init__(self, model_size="tiny"):
        logger.info(f"Loading STT Model: {model_size}")
        # Use compute_type="int8" for quantization as requested in Phase 1
        # It'll run on CPU/CUDA depending on availability
        self.model = WhisperModel(model_size, device="auto", compute_type="int8")
        logger.info("STT Model loaded successfully.")

    async def transcribe_base64(self, base64_audio: str) -> str:
        """
        Transcribes base64 encoded audio data.
        Assumes the audio is decodable by Whisper (e.g. WAV, MP3).
        """
        try:
            audio_bytes = base64.b64decode(base64_audio)
            audio_io = io.BytesIO(audio_bytes)
            
            # Using faster-whisper's transcribe method
            segments, info = self.model.transcribe(audio_io, beam_size=1)
            
            text = " ".join([segment.text for segment in segments])
            return text.strip()
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            return ""
