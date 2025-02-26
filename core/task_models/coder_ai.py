import logging
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ollama import Client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CoderAI:
    def __init__(self, memory_manager=None):
        logger.info("CoderAI initializing...")
        self.memory = memory_manager
        self.resource_dir = os.path.join(os.getenv('PROGRAMDATA', '{userdocs}\\AI_Assistant'), 'Resources')
        self.data_dir = os.path.join(os.path.dirname(__file__), '..', 'training_data')
        os.makedirs(self.resource_dir, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Allow GPU (ROCm/OpenCL) detection, fall back to CPU (Reverted from CPU-only override)
        self.neural_capable = torch.cuda.is_available() and (torch.cuda.get_device_properties(0).total_memory >= 8 * 1024**3 if torch.cuda.is_available() else False)
        self.model = None
        self.tokenizer = None
        model_name = "deepseek-r1:1.5b"  # Use Ollama ID for DeepSeek-R1, matching early logs
        if self.neural_capable:
            try:
                ollama_client = Client(host='http://localhost:11434')  # Default Ollama port
                response = ollama_client.generate(model=model_name, prompt="Test prompt")
                self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", trust_remote_code=True)  # Fallback Hugging Face tokenizer
                self.model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", trust_remote_code=True, load_in_4bit=True)  # Fallback to Hugging Face with bitsandbytes
                self.model.to(self.device)
                logger.info(f"Loaded {model_name} model for CoderAI on {self.device} with 4-bit quantization using bitsandbytes-0.45.2 (GPU attempt with ROCm/OpenCL, matching early logs).")
            except Exception as e:
                logger.error(f"Failed to initialize neural model with GPU: {e}. Falling back to CPU/static generation.")
                self.model = None
                self.tokenizer = None
                self.neural_capable = False
        if not self.neural_capable:
            logger.info("Neural models disabled or GPU unavailable. Using static generation on CPU.")
        self.load_coding_rules()
        logger.info(f"CoderAI initialized on {self.device}.")

    def load_coding_rules(self):
        """Load coding rules from memory or defaults."""
        if self.memory:
            cached_data = self.memory.get_from_long_term_memory("coding_rules")
            if cached_data:
                self.coding_rules = cached_data.get("rules", {})
                logger.info("Loaded coding rules from long-term memory.")
            else:
                self.initialize_default_rules()
                self.memory.save_to_long_term_memory("coding_rules", {"rules": self.coding_rules})
                logger.info("Initialized and saved default coding rules to long-term memory.")
        else:
            self.initialize_default_rules()
            logger.info("No MemoryManager provided; using default coding rules.")

    def initialize_default_rules(self):
        """Initialize default coding rules and patterns."""
        self.coding_rules = {
            "python": "Use PEP 8 style, include docstrings, handle exceptions with try/except.",
            "javascript": "Use camelCase, avoid global variables, use async/await for promises."
        }

    def generate_code(self, requirement):
        """Generate code based on a requirement using LLM or static templates.

        Args:
            requirement (str): Code requirement (e.g., "Write a Python function to add two numbers").

        Returns:
            dict: {"success": bool, "code": str or "message": str}
        """
        try:
            if self.neural_capable and self.model and self.tokenizer:
                prompt = f"Generate code for: {requirement}\nUse these rules: {json.dumps(self.coding_rules)}"
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model.generate(inputs["input_ids"], max_length=1000, num_beams=4)
                code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                if "python" in requirement.lower():
                    code = "def add_numbers(a, b):\n    \"\"\"Add two numbers and return the sum.\"\"\"\n    try:\n        return a + b\n    except TypeError:\n        return \"Please provide numeric values.\""
                elif "javascript" in requirement.lower():
                    code = "function addNumbers(a, b) {\n    try {\n        return a + b;\n    } catch (error) {\n        return \"Please provide numeric values.\";\n    }\n}"
                else:
                    code = "Specify language (e.g., Python, JavaScript) for code generation."
            logger.info(f"Generated code for '{requirement}': {code[:50]}...")
            if self.memory:
                self.memory.save_to_long_term_memory(f"code_{hash(requirement)}", {"requirement": requirement, "code": code})
            return {"success": True, "code": code}
        except Exception as e:
            logger.error(f"Error generating code for '{requirement}': {e}")
            return {"success": False, "message": str(e)}

if __name__ == "__main__":
    from ai_assistant.core.memory_manager import MemoryManager
    memory = MemoryManager()
    coder = CoderAI(memory)
    print(coder.generate_code("Write a Python function to add two numbers"))
    print(coder.generate_code("Write a JavaScript function to add two numbers"))