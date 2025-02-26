import logging
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Force CPU mode for RX 580 on Windows
torch.cuda.is_available = lambda: False
device = torch.device('cpu')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BusinessManager:
    def __init__(self, ai_engine=None, memory_manager=None, model_name=None):
        logger.info("BusinessManager initializing...")
        self.ai_engine = ai_engine
        self.memory = memory_manager
        self.resource_dir = os.path.join(os.getenv('PROGRAMDATA', '{userdocs}\\AI_Assistant'), 'Resources')
        self.data_dir = os.path.join(os.path.dirname(__file__), '..', 'training_data')
        os.makedirs(self.resource_dir, exist_ok=True)
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
        self.load_business_data()
        logger.info("BusinessManager initialized.")

    def load_business_data(self):
        """Load static business data from JSON file."""
        kb_path = os.path.join(self.data_dir, 'business_mgr_kb.json')
        try:
            with open(kb_path, 'r') as f:
                self.kb = json.load(f)
                logger.info("Loaded business manager knowledge base.")
        except Exception as e:
            logger.error(f"Failed to load business_mgr_kb.json: {e}. Using default.")
            self.kb = {
                "strategies": {
                    "cost_reduction": "Reduce overhead costs, optimize supply chain.",
                    "profit_increase": "Upsell products, expand market reach."
                }
            }

    def analyze_strategy(self, request):
        """Analyze business strategy and suggest improvements.

        Args:
            request (str): Query (e.g., "Improve profitability for a small store").

        Returns:
            dict: {"success": bool, "advice": str or "message": str}
        """
        try:
            if self.neural_capable and self.model and self.tokenizer:
                prompt = f"As a business manager, suggest improvements for: {request}"
                inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = self.model.generate(inputs["input_ids"], max_length=500, num_beams=4)
                advice = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                if "profit" in request.lower():
                    advice = self.kb["strategies"]["profit_increase"]
                elif "cost" in request.lower():
                    advice = self.kb["strategies"]["cost_reduction"]
                else:
                    advice = "Specify 'profit' or 'cost' for targeted advice."
            logger.info(f"Analyzed strategy for '{request}': {advice}")
            if self.memory:
                self.memory.save_to_long_term_memory(f"business_{hash(request)}", {"request": request, "advice": advice})
            return {"success": True, "advice": advice}
        except Exception as e:
            logger.error(f"Error analyzing strategy for '{request}': {e}")
            return {"success": False, "message": f"Error: {e}"}

if __name__ == "__main__":
    from ai_assistant.core.memory_manager import MemoryManager
    from ai_assistant.core.ai_engine import AIEngine
    memory = MemoryManager()
    ai_engine = AIEngine(model_name="deepseek-r1:1.5b")
    mgr = BusinessManager(ai_engine, memory)
    print(mgr.analyze_strategy("Improve profitability for a small store"))