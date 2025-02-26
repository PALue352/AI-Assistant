import logging
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ollama import Client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageAndVideoDecoder:
    def __init__(self, ai_engine=None, memory_manager=None):
        logger.info("ImageAndVideoDecoder initializing...")
        self.ai_engine = ai_engine
        self.memory = memory_manager
        self.resource_dir = os.path.join(os.getenv('PROGRAMDATA', '{userdocs}\\AI_Assistant'), 'Resources')
        self.data_dir = os.path.join(os.path.dirname(__file__), '..', 'training_data')
        os.makedirs(self.resource_dir, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Allow GPU (ROCm/OpenCL) detection, fall back to CPU (Reverted from CPU-only override)
        self.neural_capable = torch.cuda.is_available() and (torch.cuda.get_device_properties(0).total_memory >= 8 * 1024**3 if torch.cuda.is_available() else False)
        self.model = None
        self.tokenizer = None
        model_name = self.ai_engine.model_name if self.ai_engine else "deepseek-r1:1.5b"  # Use Ollama ID for DeepSeek-R1, matching early logs
        if self.neural_capable:
            try:
                ollama_client = Client(host='http://localhost:11434')  # Default Ollama port
                response = ollama_client.generate(model=model_name, prompt="Test prompt")
                self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", trust_remote_code=True)  # Fallback Hugging Face tokenizer
                self.model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", trust_remote_code=True, load_in_4bit=True)  # Fallback to Hugging Face with bitsandbytes
                self.model.to(self.device)
                logger.info(f"Loaded {model_name} model for ImageAndVideoDecoder on {self.device} with 4-bit quantization using bitsandbytes-0.45.2 (GPU attempt with ROCm/OpenCL, matching early logs).")
            except Exception as e:
                logger.error(f"Failed to initialize neural model with GPU: {e}. Falling back to CPU/static decoding.")
                self.model = None
                self.tokenizer = None
                self.neural_capable = False
        if not self.neural_capable:
            logger.info("Neural models disabled. Using static decoding on CPU.")
        self.load_decoder_knowledge()
        logger.info(f"ImageAndVideoDecoder initialized on {self.device}.")

    def load_decoder_knowledge(self):
        """Load decoder knowledge base from JSON file or memory."""
        kb_path = os.path.join(self.data_dir, 'decoder_kb.json')
        try:
            with open(kb_path, 'r') as f:
                self.decoder_data = json.load(f)
                logger.info("Loaded decoder knowledge base from file.")
        except FileNotFoundError:
            logger.warning(f"Decoder knowledge base not found at {kb_path}. Using default data.")
            self.decoder_data = {"image_format": "JPEG, PNG for images; MP4, AVI for videos.", "encoding": "H.264, H.265 for video compression."}
        except Exception as e:
            logger.error(f"Failed to load decoder_kb.json: {e}. Using default data.")
            self.decoder_data = {"image_format": "JPEG, PNG for images; MP4, AVI for videos.", "encoding": "H.264, H.265 for video compression."}
        if self.memory:
            cached = self.memory.get_from_long_term_memory("decoder_cache")
            if cached:
                self.decoder_data.update(cached.get("data", {}))
                logger.info("Updated decoder data from long-term memory.")

    def decode_media(self, media_description):
        """Decode media (image/video) information using LLM or static templates.

        Args:
            media_description (str): Description of media (e.g., "Decode JPEG image format").

        Returns:
            dict: {"success": bool, "decoding": str or "message": str}
        """
        try:
            if self.neural_capable and self.model and self.tokenizer:
                prompt = f"Decode media: {media_description}\nUse this knowledge: {json.dumps(self.decoder_data)}"
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model.generate(inputs["input_ids"], max_length=500, num_beams=4)
                decoding = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                if "jpeg" in media_description.lower() or "png" in media_description.lower():
                    decoding = self.decoder_data["image_format"]
                elif "mp4" in media_description.lower() or "avi" in media_description.lower():
                    decoding = self.decoder_data["encoding"]
                else:
                    decoding = "Cannot decode; specify image (JPEG, PNG) or video (MP4, AVI) format."
            logger.info(f"Decoded media '{media_description}': {decoding[:50]}...")
            if self.memory:
                self.memory.save_to_long_term_memory(f"decode_{hash(media_description)}", {"description": media_description, "decoding": decoding})
            return {"success": True, "decoding": decoding}
        except Exception as e:
            logger.error(f"Error decoding media '{media_description}': {e}")
            return {"success": False, "message": str(e)}

if __name__ == "__main__":
    from ai_assistant.core.memory_manager import MemoryManager
    from ai_assistant.core.ai_engine import AIEngine
    memory = MemoryManager()
    ai_engine = AIEngine(model_name="deepseek-r1:1.5b")
    decoder = ImageAndVideoDecoder(ai_engine, memory)
    print(decoder.decode_media("Decode JPEG image format"))
    print(decoder.decode_media("Decode MP4 video format"))