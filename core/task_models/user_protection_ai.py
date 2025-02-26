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

class UserProtectionAI:
    def __init__(self, ai_engine=None, memory_manager=None, model_name=None):
        logger.info("UserProtectionAI initializing...")
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
        self.load_protection_data()
        logger.info("UserProtectionAI initialized.")

    def load_protection_data(self):
        """Load static protection data from JSON file."""
        kb_path = os.path.join(self.data_dir, 'protection_kb.json')
        try:
            with open(kb_path, 'r') as f:
                self.kb = json.load(f)
                logger.info("Loaded protection knowledge base.")
        except Exception as e:
            logger.error(f"Failed to load protection_kb.json: {e}. Using default.")
            self.kb = {
                "threats": {
                    "phishing": {
                        "signs": ["Suspicious URLs", "Urgent language", "Unknown sender"],
                        "action": "Do not click links; verify sender; report to IT."
                    },
                    "malware": {
                        "signs": ["Slow system", "Pop-ups", "Unusual activity"],
                        "action": "Run antivirus scan; disconnect from internet; seek expert help."
                    }
                }
            }

    def assess_threat(self, situation):
        """Assess a situation for potential threats and advise action.

        Args:
            situation (str): User situation (e.g., "Received a suspicious email", "System is slow").

        Returns:
            dict: {"success": bool, "threat": str, "advice": str or "message": str}
        """
        try:
            if self.neural_capable and self.model and self.tokenizer:
                prompt = f"As a user protection expert, assess this situation and advise: {situation}"
                inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = self.model.generate(inputs["input_ids"], max_length=500, num_beams=4)
                advice = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                threat = "Potential threat identified" if "threat" in advice.lower() else "No clear threat"
            else:
                threat = "Unknown"
                advice = "Unable to assess fully without neural model."
                for key, value in self.kb["threats"].items():
                    if any(sign.lower() in situation.lower() for sign in value["signs"]):
                        threat = key.capitalize()
                        advice = value["action"]
                        break
                if threat == "Unknown":
                    advice = "No known threat detected; monitor and report unusual activity."
            logger.info(f"Assessed '{situation}': Threat={threat}, Advice={advice}")
            if self.memory:
                self.memory.save_to_long_term_memory(f"protection_{hash(situation)}", {"situation": situation, "threat": threat, "advice": advice})
            return {"success": True, "threat": threat, "advice": advice}
        except Exception as e:
            logger.error(f"Error assessing '{situation}': {e}")
            return {"success": False, "message": f"Error: {e}"}

if __name__ == "__main__":
    from ai_assistant.core.memory_manager import MemoryManager
    from ai_assistant.core.ai_engine import AIEngine
    memory = MemoryManager()
    ai_engine = AIEngine(model_name="deepseek-r1:1.5b")
    protect_ai = UserProtectionAI(ai_engine, memory)
    print(protect_ai.assess_threat("Received a suspicious email"))
    print(protect_ai.assess_threat("System is slow"))