import logging
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ollama import Client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MotionAI:
    def __init__(self, ai_engine=None, memory_manager=None):
        logger.info("MotionAI initializing...")
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
                logger.info(f"Loaded {model_name} model for MotionAI on {self.device} with 4-bit quantization using bitsandbytes-0.45.2 (GPU attempt with ROCm/OpenCL, matching early logs).")
            except Exception as e:
                logger.error(f"Failed to initialize neural model with GPU: {e}. Falling back to CPU/static strategies.")
                self.model = None
                self.tokenizer = None
                self.neural_capable = False
        if not self.neural_capable:
            logger.info("Neural models disabled or GPU unavailable. Using static strategies on CPU.")
        self.load_motion_knowledge()
        logger.info(f"MotionAI initialized on {self.device}.")

    def load_motion_knowledge(self):
        """Load motion knowledge base from JSON file or memory."""
        kb_path = os.path.join(self.data_dir, 'motion_kb.json')
        try:
            with open(kb_path, 'r') as f:
                self.motion_data = json.load(f)
                logger.info("Loaded motion knowledge base from file.")
        except FileNotFoundError:
            logger.warning(f"Motion knowledge base not found at {kb_path}. Using default data.")
            self.motion_data = {"velocity": "v = u + at", "acceleration": "a = (v - u) / t"}
        except Exception as e:
            logger.error(f"Failed to load motion_kb.json: {e}. Using default data.")
            self.motion_data = {"velocity": "v = u + at", "acceleration": "a = (v - u) / t"}
        if self.memory:
            cached = self.memory.get_from_long_term_memory("motion_cache")
            if cached:
                self.motion_data.update(cached.get("data", {}))
                logger.info("Updated motion data from long-term memory.")

    def analyze_motion(self, scenario):
        """Analyze motion scenario using LLM or static templates.

        Args:
            scenario (str): Motion scenario description (e.g., "Object moving with initial velocity 5m/s, acceleration 2m/s², time 3s").

        Returns:
            dict: {"success": bool, "analysis": str or "message": str}
        """
        try:
            if self.neural_capable and self.model and self.tokenizer:
                prompt = f"Analyze motion scenario: {scenario}\nUse this knowledge: {json.dumps(self.motion_data)}"
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model.generate(inputs["input_ids"], max_length=500, num_beams=4)
                analysis = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                if "velocity" in scenario.lower() and "initial velocity" in scenario.lower() and "acceleration" in scenario.lower() and "time" in scenario.lower():
                    u = float(scenario.split("initial velocity")[1].split("m/s")[0]) if "m/s" in scenario else 0
                    a = float(scenario.split("acceleration")[1].split("m/s²")[0]) if "m/s²" in scenario else 0
                    t = float(scenario.split("time")[1].split("s")[0]) if "s" in scenario else 0
                    analysis = f"Velocity: v = {u} + {a} * {t} = {u + a * t} m/s"
                else:
                    analysis = "Cannot analyze; specify velocity, acceleration, and time with units."
            logger.info(f"Analyzed motion scenario '{scenario}': {analysis[:50]}...")
            if self.memory:
                self.memory.save_to_long_term_memory(f"motion_{hash(scenario)}", {"scenario": scenario, "analysis": analysis})
            return {"success": True, "analysis": analysis}
        except Exception as e:
            logger.error(f"Error analyzing motion scenario '{scenario}': {e}")
            return {"success": False, "message": str(e)}

if __name__ == "__main__":
    from ai_assistant.core.memory_manager import MemoryManager
    from ai_assistant.core.ai_engine import AIEngine
    memory = MemoryManager()
    ai_engine = AIEngine(model_name="deepseek-r1:1.5b")
    motion_ai = MotionAI(ai_engine, memory)
    print(motion_ai.analyze_motion("Object moving with initial velocity 5m/s, acceleration 2m/s², time 3s"))