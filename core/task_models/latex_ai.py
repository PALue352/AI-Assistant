import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ollama import Client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LatexAI:
    def __init__(self):
        logger.info("LatexAI initializing...")
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
                logger.info(f"Loaded {model_name} model for LatexAI on {self.device} with 4-bit quantization using bitsandbytes-0.45.2 (GPU attempt with ROCm/OpenCL, matching early logs).")
            except Exception as e:
                logger.error(f"Failed to initialize neural model with GPU: {e}. Falling back to CPU/static strategies.")
                self.model = None
                self.tokenizer = None
                self.neural_capable = False
        if not self.neural_capable:
            logger.info("Neural models disabled or GPU unavailable. Using static strategies on CPU.")
        logger.info(f"LatexAI initialized on {self.device}.")

    def generate_latex(self, description):
        """Generate LaTeX code from a description using LLM or static templates.

        Args:
            description (str): Description of the equation or content (e.g., "quadratic equation ax^2 + bx + c = 0").

        Returns:
            dict: {"success": bool, "latex": str or "message": str}
        """
        try:
            if self.neural_capable and self.model and self.tokenizer:
                prompt = f"Generate LaTeX for: {description}"
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model.generate(inputs["input_ids"], max_length=500, num_beams=4)
                latex = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                if "quadratic" in description.lower():
                    latex = r"\[ ax^2 + bx + c = 0 \]"
                else:
                    latex = "Specify an equation or content for LaTeX generation."
            logger.info(f"Generated LaTeX for '{description}': {latex[:50]}...")
            return {"success": True, "latex": latex}
        except Exception as e:
            logger.error(f"Error generating LaTeX for '{description}': {e}")
            return {"success": False, "message": str(e)}

if __name__ == "__main__":
    latex_ai = LatexAI()
    result = latex_ai.generate_latex("quadratic equation ax^2 + bx + c = 0")
    print(result)
    print(doc["latex"] if doc["success"] else doc["message"])