import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Force CPU mode for RX 580 on Windows
torch.cuda.is_available = lambda: False
device = torch.device('cpu')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PublicRelationsAI:
    def __init__(self, ai_engine=None, memory_manager=None, model_name=None):
        logger.info("PublicRelationsAI initializing...")
        self.ai_engine = ai_engine
        self.memory = memory_manager
        self.neural_capable = torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= 8 * 1024**3
        self.model = None
        self.tokenizer = None
        model_name = model_name or (self.ai_engine.model_name if self.ai_engine else "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        if self.neural_capable:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model.to(device)  # Force CPU for RX 580
                logger.info(f"Neural model {model_name} initialized with 4-bit quantization on CPU.")
            except Exception as e:
                logger.error(f"Failed to initialize neural model: {e}. Using static strategies.")
                self.model = None
                self.tokenizer = None
        else:
            logger.info("Neural models disabled. Using static strategies.")

    def manage_pr(self, request):
        try:
            if self.neural_capable and self.model and self.tokenizer:
                prompt = f"As a PR expert, process: {request}"
                inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = self.model.generate(inputs["input_ids"], max_length=500, num_beams=4)
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                response = "Static response—neural model not available."
            logger.info(f"Managed PR for '{request}': {response}")
            if self.memory:
                self.memory.save_to_long_term_memory(f"pr_{hash(request)}", {"request": request, "response": response})
            return {"success": True, "response": response}
        except Exception as e:
            logger.error(f"Error managing PR: {e}")
            return {"success": False, "message": f"Error: {e}"}

if __name__ == "__main__":
    from ai_assistant.core.memory_manager import MemoryManager
    from ai_assistant.core.ai_engine import AIEngine
    memory = MemoryManager()
    ai_engine = AIEngine(model_name="deepseek-r1:1.5b")
    pr_ai = PublicRelationsAI(ai_engine, memory)
    print(pr_ai.manage_pr("Create a PR strategy for a product launch"))