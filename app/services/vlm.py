import logging
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import base64
import io

logger = logging.getLogger(__name__)

class VLMService:
    def __init__(self, model_id="llava-hf/llava-1.5-7b-hf", mock=True):
        self.mock = mock
        if self.mock:
            logger.info("Initializing Mock VLM Service for fast testing.")
            return

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading VLM {model_id} on {self.device}")
        
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
        self.model.to(self.device)
        logger.info("VLM loaded successfully.")

    async def generate(self, prompt: str, image_b64: str) -> str:
        if self.mock:
            return f"[Mock VLM Response]: I see the image. Regarding '{prompt}', it looks great."

        try:
            image_bytes = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # Format prompt based on specific VLM requirements (e.g., LLaVA)
            text_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
            
            inputs = self.processor(text=text_prompt, images=image, return_tensors="pt").to(self.device)
            
            # Generate
            generate_ids = self.model.generate(**inputs, max_new_tokens=128)
            response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            
            # Basic cleanup if assistant tag remains
            if "ASSISTANT:" in response:
                response = response.split("ASSISTANT:")[1].strip()
            
            return response
        except Exception as e:
            logger.error(f"Error during VLM inference: {e}")
            return "Sorry, I couldn't process that image."
