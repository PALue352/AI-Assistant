import logging
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ollama import Client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HowToThinkAI:
    def __init__(self, ai_engine=None, memory_manager=None):
        logger.info("HowToThinkAI initializing...")
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
                logger.info(f"Loaded {model_name} model for HowToThinkAI on {self.device} with 4-bit quantization using bitsandbytes-0.45.2 (GPU attempt with ROCm/OpenCL, matching early logs).")
            except Exception as e:
                logger.error(f"Failed to initialize neural model with GPU: {e}. Falling back to CPU/static strategies or no model specified.")
                self.model = None
                self.tokenizer = None
                self.neural_capable = False
        if not self.neural_capable:
            logger.info("Neural models disabled or no model specified. Using static strategies or no model specified on CPU.")
        self.load_thinking_strategies()
        logger.info(f"HowToThinkAI initialized on {self.device}.")

    def load_thinking_strategies(self):
        """Load thinking strategies from memory or defaults."""
        if self.memory:
            cached_data = self.memory.get_from_long_term_memory("thinking_cache")
            if cached_data:
                self.thinking_strategies = cached_data.get("strategies", {})
                logger.info("Loaded thinking strategies from long-term memory.")
            else:
                self.initialize_default_strategies()
                self.memory.save_to_long_term_memory("thinking_cache", {"strategies": self.thinking_strategies})
                logger.info("Initialized and saved default thinking strategies to long-term memory.")
        else:
            self.initialize_default_strategies()
            logger.info("No MemoryManager provided; using default thinking strategies.")

    def initialize_default_strategies(self):
        """Initialize default thinking strategies and patterns."""
        self.thinking_strategies = {
            "logical": "Use deductive reasoning: Start with general principles, derive specific conclusions.",
            "creative": "Brainstorm ideas freely, connect unrelated concepts to form novel solutions."
        }

    def process_thinking_task(self, task):
        """Process a thinking task using LLM or static templates.

        Args:
            task (str): Thinking task description (e.g., "How to solve a complex problem logically?").

        Returns:
            dict: {"success": bool, "strategy": str or "message": str}
        """
        try:
            if self.neural_capable and self.model and self.tokenizer:
                prompt = f"Provide a thinking strategy for: {task}\nUse these strategies: {json.dumps(self.thinking_strategies)}"
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model.generate(inputs["input_ids"], max_length=500, num_beams=4)
                strategy = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                if "logically" in task.lower() or "logical" in task.lower():
                    strategy = self.thinking_strategies["logical"]
                elif "creatively" in task.lower() or "creative" in task.lower():
                    strategy = self.thinking_strategies["creative"]
                else:
                    strategy = "Use logical or creative thinking based on the problem; specify approach for best results."
            logger.info(f"Processed thinking task '{task}': {strategy[:50]}...")
            if self.memory:
                self.memory.save_to_long_term_memory(f"thinking_{hash(task)}", {"task": task, "strategy": strategy})
            return {"success": True, "strategy": strategy}
        except Exception as e:
            logger.error(f"Error processing thinking task '{task}': {e}")
            return {"success": False, "message": str(e)}

if __name__ == "__main__":
    from ai_assistant.core.memory_manager import MemoryManager
    from ai_assistant.core.ai_engine import AIEngine
    memory = MemoryManager()
    ai_engine = AIEngine(model_name="deepseek-r1:1.5b")
    thinker = HowToThinkAI(ai_engine, memory)
    print(thinker.process_thinking_task("How to solve a complex problem logically?"))
    print(thinker.process_thinking_task("How to generate creative ideas for a project?"))