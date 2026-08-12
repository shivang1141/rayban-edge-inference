import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, model_id="Qwen/Qwen1.5-0.5B-Chat", mock=True):
        self.mock = mock
        if self.mock:
            logger.info("Initializing Mock LLM Service for fast testing.")
            return

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading LLM {model_id} on {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
        self.model.to(self.device)
        logger.info("LLM loaded successfully.")

    async def generate(self, prompt: str) -> str:
        if self.mock:
            return f"[Mock LLM Response]: You asked '{prompt}'. This is a simulated response."

        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=128,
            pad_token_id=self.tokenizer.eos_token_id
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()
