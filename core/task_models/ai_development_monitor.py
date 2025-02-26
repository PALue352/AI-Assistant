import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Force CPU mode for RX 580 on Windows
torch.cuda.is_available = lambda: False
device = torch.device('cpu')

logger = logging.getLogger(__name__)

class AIDevelopmentMonitor:
    def __init__(self):
        logger.info("AIDevelopmentMonitor initializing...")
        self.model_name = "Qwen/Qwen-7B"  # Adjust as needed for monitoring AI development
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True)
        self.model.to(device)
        logger.info("AIDevelopmentMonitor initialized on CPU.")

    def monitor_development(self, project_data):
        """Monitors AI development progress or issues."""
        try:
            inputs = self.tokenizer(f"Analyze AI development progress: {project_data}", return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=200, num_beams=4)
            analysis = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return {"success": True, "analysis": analysis}
        except Exception as e:
            logger.error(f"Error monitoring AI development: {e}")
            return {"success": False, "message": str(e)}