import logging
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ollama import Client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketingManager:
    def __init__(self, ai_engine=None, memory_manager=None):
        logger.info("MarketingManager initializing...")
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
                logger.info(f"Loaded {model_name} model for MarketingManager on {self.device} with 4-bit quantization using bitsandbytes-0.45.2 (GPU attempt with ROCm/OpenCL, matching early logs).")
            except Exception as e:
                logger.error(f"Failed to initialize neural model with GPU: {e}. Falling back to CPU/static strategies.")
                self.model = None
                self.tokenizer = None
                self.neural_capable = False
        if not self.neural_capable:
            logger.info("Neural models disabled or GPU unavailable. Using static strategies on CPU.")
        self.load_marketing_knowledge()
        logger.info(f"MarketingManager initialized on {self.device}.")

    def load_marketing_knowledge(self):
        """Load marketing knowledge base from JSON file or memory."""
        kb_path = os.path.join(self.data_dir, 'marketing_kb.json')
        try:
            with open(kb_path, 'r') as f:
                self.marketing_data = json.load(f)
                logger.info("Loaded marketing knowledge base from file.")
        except FileNotFoundError:
            logger.warning(f"Marketing knowledge base not found at {kb_path}. Using default data.")
            self.marketing_data = {"strategy": "Target audience analysis, SEO optimization, social media campaigns."}
        except Exception as e:
            logger.error(f"Failed to load marketing_kb.json: {e}. Using default data.")
            self.marketing_data = {"strategy": "Target audience analysis, SEO optimization, social media campaigns."}
        if self.memory:
            cached = self.memory.get_from_long_term_memory("marketing_cache")
            if cached:
                self.marketing_data.update(cached.get("data", {}))
                logger.info("Updated marketing data from long-term memory.")

    def develop_marketing_strategy(self, product):
        """Develop a marketing strategy for a product using LLM or static templates.

        Args:
            product (str): Product description (e.g., "Eco-friendly water bottle").

        Returns:
            dict: {"success": bool, "strategy": str or "message": str}
        """
        try:
            if self.neural_capable and self.model and self.tokenizer:
                prompt = f"Develop a marketing strategy for: {product}\nUse this knowledge: {json.dumps(self.marketing_data)}"
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model.generate(inputs["input_ids"], max_length=500, num_beams=4)
                strategy = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                strategy = f"Marketing strategy for {product}: {self.marketing_data['strategy']} Focus on eco-conscious consumers."
            logger.info(f"Developed marketing strategy for '{product}': {strategy[:50]}...")
            if self.memory:
                self.memory.save_to_long_term_memory(f"marketing_{hash(product)}", {"product": product, "strategy": strategy})
            return {"success": True, "strategy": strategy}
        except Exception as e:
            logger.error(f"Error developing marketing strategy for '{product}': {e}")
            return {"success": False, "message": str(e)}

if __name__ == "__main__":
    from ai_assistant.core.memory_manager import MemoryManager
    from ai_assistant.core.ai_engine import AIEngine
    memory = MemoryManager()
    ai_engine = AIEngine(model_name="deepseek-r1:1.5b")
    marketing = MarketingManager(ai_engine, memory)
    print(marketing.develop_marketing_strategy("Eco-friendly water bottle"))