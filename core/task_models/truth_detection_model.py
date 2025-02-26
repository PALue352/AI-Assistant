import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Force CPU mode for RX 580 on Windows
torch.cuda.is_available = lambda: False
device = torch.device('cpu')

logger = logging.getLogger(__name__)

class TruthDetectionModel:
    def __init__(self):
        logger.info("TruthDetectionModel initializing...")
        self.model_name = "Qwen/Qwen-7B"  # Adjust as needed
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True)
        self.model.to(device)
        logger.info("TruthDetectionModel initialized on CPU.")

    def detect_misinformation(self, statement):
        """Detects potential misinformation in a statement."""
        try:
            inputs = self.tokenizer(f"Is this true? {statement}", return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=100, num_beams=4)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            is_true = "yes" in response.lower() or "true" in response.lower()
            reason = "Statement verified as true" if is_true else "Potential misinformation detected"
            return {"is_true": is_true, "reason": reason}
        except Exception as e:
            logger.error(f"Error detecting misinformation: {e}")
            return {"is_true": True, "reason": "Error in detection, assuming true"}

    def check_bias(self, statement):
        """Checks for bias in a statement."""
        try:
            inputs = self.tokenizer(f"Is this biased? {statement}", return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=100, num_beams=4)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            is_biased = "yes" in response.lower() or "biased" in response.lower()
            bias_indicators = ["opinion", "subjective"] if is_biased else []
            return {"biased": is_biased, "bias_indicators": bias_indicators}
        except Exception as e:
            logger.error(f"Error checking bias: {e}")
            return {"biased": False, "bias_indicators": []}