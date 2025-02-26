# common_sense_model.py (v1.0.3)
import logging
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Get the root logger (configured in overseer.py)
logger = logging.getLogger(__name__)

class CommonSenseModel:
    def __init__(self, overseer):
        logger.info("CommonSenseModel initializing...")
        self.ai_engine = overseer.ai_engine
        self.resource_dir = os.path.join(os.getenv('PROGRAMDATA', '{userdocs}\\AI_Assistant'), 'Resources')
        self.data_dir = os.path.join(os.path.dirname(__file__), '..', 'training_data')
        os.makedirs(self.resource_dir, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Allow GPU (ROCm/OpenCL) detection, fall back to CPU (Reverted from CPU-only override)
        self.neural_capable = torch.cuda.is_available() and (torch.cuda.get_device_properties(0).total_memory >= 8 * 1024**3 if torch.cuda.is_available() else False)
        self.model = None
        self.tokenizer = None
        #Removed model name from parameter
        if self.neural_capable:
            try:
                # Use the current ai_engine model
                self.tokenizer = self.ai_engine.tokenizer  
                self.model = self.ai_engine.model
                logger.info(f"Loaded model for CommonSenseModel on {self.device} with 4-bit quantization using bitsandbytes-0.45.2 (GPU attempt with ROCm/OpenCL, matching early logs).")
            except Exception as e:
                logger.error(f"Failed to initialize neural model with GPU: {e}. Falling back to CPU/static strategies.")
                self.model = None
                self.tokenizer = None
                self.neural_capable = False
        if not self.neural_capable:
            logger.info("Neural models disabled or GPU unavailable. Using static strategies on CPU.")
        self.load_common_sense_data()
        logger.info(f"CommonSenseModel initialized on {self.device}.")

    def load_common_sense_data(self):
        """Load common sense knowledge base from JSON file or memory."""
        kb_path = os.path.join(self.data_dir, 'common_sense_kb.json')  # Updated to correct file name
        try:
            with open(kb_path, 'r') as f:
                self.common_sense_data = json.load(f)
                logger.info("Loaded common sense knowledge base from file.")
        except FileNotFoundError:
            logger.error(f"Common sense knowledge base not found at {kb_path}. Using default data. Create 'common_sense_kb.json' at {kb_path} with default content: {{\"facts\": [\"Water boils at 100°C\", \"The Earth orbits the Sun\"]}}")
            self.common_sense_data = {"facts": ["Water boils at 100°C", "The Earth orbits the Sun"]}
        except Exception as e:
            logger.error(f"Failed to load common_sense_kb.json: {e}. Using default data.")
            self.common_sense_data = {"facts": ["Water boils at 100°C", "The Earth orbits the Sun"]}


    def apply_common_sense(self, statement):
        if self.ai_engine:
            try:
                prompt = f"Is this statement common sense? '{statement}'"
                response = self.ai_engine.process_request(prompt)
                if response and response["type"] == "text":
                     return response["content"]
                return "Could not determine."
            except Exception as e:
                logger.error(f"Error in check_common_sense: {e}")
                return "Error checking common sense."
        else:
            return "CommonSenseModel not initialized."


    def suggest_alternative(self, statement):
        if self.ai_engine:
            try:
                prompt = f"Given the statement '{statement}', suggest a more common-sense alternative."
                response = self.ai_engine.process_request(prompt)
                if response and response["type"] == "text":
                     return response["content"]
                return "Could not suggest an alternative."
            except Exception as e:
                logger.error(f"Error in suggest_alternative: {e}")
                return "Error suggesting alternative."

        else:
            return "CommonSenseModel not initialized."


if __name__ == '__main__':
    #You cannot run this file directly because Overseer needs to be initialized to run it.
    #Create dummy overseer for testing.
    class MockOverseer:
        def __init__(self):
            class MockAIEngine:
                def __init__(self):
                    self.model_name = "MockModel"
                def process_request(self, prompt):
                    return {"type": "text", "content": f"Mock response to: {prompt}"}
            self.ai_engine = MockAIEngine()

    overseer = MockOverseer()
    common_sense_model = CommonSenseModel(overseer)
    result = common_sense_model.check_common_sense("The sky is blue.")
    print(f"Check common sense: {result}")
    result = common_sense_model.suggest_alternative("The sky is green.")
    print(f"Suggest alternative: {result}")