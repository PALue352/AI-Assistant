import logging
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ollama import Client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PatternModel:
    def __init__(self, ai_engine=None, memory_manager=None, model_name=None):
        logger.info("PatternModel initializing...")
        self.ai_engine = ai_engine
        self.memory = memory_manager
        self.resource_dir = os.path.join(os.getenv('PROGRAMDATA', '{userdocs}\\AI_Assistant'), 'Resources')
        self.data_dir = os.path.join(os.path.dirname(__file__), '..', 'training_data')
        os.makedirs(self.resource_dir, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Allow GPU (ROCm/OpenCL) detection, fall back to CPU (Reverted from CPU-only override)
        self.neural_capable = torch.cuda.is_available() and (torch.cuda.get_device_properties(0).total_memory >= 8 * 1024**3 if torch.cuda.is_available() else False)
        self.model = None
        self.tokenizer = None
        model_name = model_name or (self.ai_engine.model_name if self.ai_engine else "deepseek-r1:1.5b")  # Use Ollama ID for DeepSeek-R1, matching early logs
        if self.neural_capable:
            try:
                ollama_client = Client(host='http://localhost:11434')  # Default Ollama port
                response = ollama_client.generate(model=model_name, prompt="Test prompt")
                self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", trust_remote_code=True)  # Fallback Hugging Face tokenizer
                self.model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", trust_remote_code=True, load_in_4bit=True)  # Fallback to Hugging Face with bitsandbytes
                self.model.to(self.device)
                logger.info(f"Loaded {model_name} model for PatternModel on {self.device} with 4-bit quantization using bitsandbytes-0.45.2 (GPU attempt with ROCm/OpenCL, matching early logs).")
            except Exception as e:
                logger.error(f"Failed to initialize neural model with GPU: {e}. Falling back to CPU/static strategies or no model specified.")
                self.model = None
                self.tokenizer = None
                self.neural_capable = False
        if not self.neural_capable:
            logger.info("Neural models disabled or no model specified. Using static strategies or no model specified on CPU.")
        self.load_pattern_data()
        logger.info(f"PatternModel initialized on {self.device}.")

    def load_pattern_data(self):
        """Load pattern recognition data from JSON file or memory."""
        kb_path = os.path.join(self.data_dir, 'pattern_kb.json')
        try:
            with open(kb_path, 'r') as f:
                self.patterns = json.load(f)
                logger.info("Loaded pattern knowledge base from file.")
        except FileNotFoundError:
            logger.warning(f"Pattern knowledge base not found at {kb_path}. Using default patterns. Create 'pattern_kb.json' at {kb_path} with default content: {{\"simple\": \"Identify repeating sequences or trends.\", \"complex\": \"Analyze multi-dimensional data for correlations.\"}}")
            self.patterns = {"simple": "Identify repeating sequences or trends.", "complex": "Analyze multi-dimensional data for correlations."}
        except Exception as e:
            logger.error(f"Failed to load pattern_kb.json: {e}. Using default patterns.")
            self.patterns = {"simple": "Identify repeating sequences or trends.", "complex": "Analyze multi-dimensional data for correlations."}
        if self.memory:
            cached = self.memory.get_from_long_term_memory("pattern_cache")
            if cached:
                self.patterns.update(cached.get("patterns", {}))
                logger.info("Updated patterns from long-term memory.")

    def recognize_pattern(self, data):
        """Recognize patterns in the provided data using LLM or static templates.

        Args:
            data (str or list): Data to analyze for patterns (e.g., "Sequence: 1, 2, 3, 4" or [1, 2, 3, 4]).

        Returns:
            dict: {"success": bool, "pattern": str or "message": str}
        """
        try:
            if self.neural_capable and self.model and self.tokenizer:
                prompt = f"Identify patterns in: {data}\nUse these patterns: {json.dumps(self.patterns)}"
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model.generate(inputs["input_ids"], max_length=500, num_beams=4)
                pattern = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                if isinstance(data, str) and "sequence" in data.lower():
                    pattern = self.patterns["simple"]
                elif isinstance(data, list) and len(data) > 2:
                    pattern = self.patterns["complex"]
                else:
                    pattern = "No recognizable pattern; specify sequence or multi-dimensional data."
            logger.info(f"Recognized pattern in '{data}': {pattern[:50]}...")
            if self.memory:
                self.memory.save_to_long_term_memory(f"pattern_{hash(str(data))}", {"data": data, "pattern": pattern})
            return {"success": True, "pattern": pattern}
        except Exception as e:
            logger.error(f"Error recognizing pattern in '{data}': {e}")
            return {"success": False, "message": str(e)}

if __name__ == "__main__":
    from ai_assistant.core.memory_manager import MemoryManager
    from ai_assistant.core.ai_engine import AIEngine
    memory = MemoryManager()
    ai_engine = AIEngine(model_name="deepseek-r1:1.5b")
    pattern_model = PatternModel(ai_engine, memory)
    print(pattern_model.recognize_pattern("Sequence: 1, 2, 3, 4"))
    print(pattern_model.recognize_pattern([1, 2, 3, 4, 5, 6]))