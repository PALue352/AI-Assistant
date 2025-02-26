import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from ollama import Client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MedicalAI:
    def __init__(self, overseer, ai_engine=None, memory_manager=None, model_name=None):
        logger.info("MedicalAI initializing...")
        self.overseer = overseer
        self.ai_engine = ai_engine
        self.memory = memory_manager
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Allow GPU (ROCm/OpenCL) detection, fall back to CPU (Reverted from CPU-only override)
        model_name = model_name or (self.ai_engine.model_name if self.ai_engine else "deepseek-r1:1.5b")  # Use Ollama ID for DeepSeek-R1, matching early logs
        self.neural_capable = torch.cuda.is_available() and (torch.cuda.get_device_properties(0).total_memory >= 8 * 1024**3 if torch.cuda.is_available() else False)
        self.tokenizer = None
        self.model = None
        if self.neural_capable:
            try:
                ollama_client = Client(host='http://localhost:11434')  # Default Ollama port
                response = ollama_client.generate(model=model_name, prompt="Test prompt")
                self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", trust_remote_code=True)  # Fallback Hugging Face tokenizer
                self.model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", trust_remote_code=True, load_in_4bit=True)  # Fallback to Hugging Face with bitsandbytes
                self.model.to(self.device)
                logger.info(f"Loaded {model_name} model for MedicalAI on {self.device} with 4-bit quantization using bitsandbytes-0.45.2 (GPU attempt with ROCm/OpenCL, matching early logs).")
            except Exception as e:
                logger.error(f"Failed to initialize neural model with GPU: {e}. Falling back to CPU/static templates.")
                self.model = None
                self.tokenizer = None
                self.neural_capable = False
        if not self.neural_capable:
            logger.warning("Neural capabilities disabled or GPU unavailable. Using static templates on CPU.")
        logger.info(f"MedicalAI initialized on {self.device}.")

    def diagnose(self, symptoms):
        """Diagnoses medical conditions based on symptoms using LLM or templates."""
        try:
            if self.neural_capable:
                inputs = self.tokenizer(f"Diagnose based on symptoms: {symptoms}", return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_length=200, num_beams=4)
                diagnosis = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return {"success": True, "diagnosis": diagnosis}
            else:
                # Static template for non-neural mode
                return {"success": True, "diagnosis": "Static diagnosis: Consult a doctor for symptoms - " + symptoms}
        except Exception as e:
            logger.error(f"Error diagnosing: {e}")
            return {"success": False, "message": str(e)}

if __name__ == "__main__":
    from ai_assistant.core.memory_manager import MemoryManager
    from ai_assistant.core.ai_engine import AIEngine
    memory = MemoryManager()
    ai_engine = AIEngine(model_name="deepseek-r1:1.5b")
    medical_ai = MedicalAI(None, ai_engine, memory)  # Overseer placeholder for testing
    print(medical_ai.diagnose("Persistent headache and nausea"))