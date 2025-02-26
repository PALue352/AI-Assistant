# ai_assistant/core/task_models/ai_hardware_manager.py
import logging
import os
import json
import torch
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AIHardwareManager:
    def __init__(self, memory_manager=None, model_name="Qwen/Qwen-1_8B"):
        logger.info("AIHardwareManager initializing...")
        self.memory = memory_manager
        self.resource_dir = os.path.join(os.getenv('PROGRAMDATA', '{userdocs}\\AI_Assistant'), 'Resources')
        self.data_dir = os.path.join(os.path.dirname(__file__), '..', 'training_data')
        os.makedirs(self.resource_dir, exist_ok=True)
        self.neural_capable = torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= 8 * 1024**3
        self.model = None
        self.tokenizer = None
        if self.neural_capable:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model.to("cuda" if torch.cuda.is_available() else "cpu")
                logger.info(f"Neural model {model_name} initialized with 4-bit quantization.")
            except Exception as e:
                logger.error(f"Failed to initialize neural model: {e}. Using static monitoring.")
                self.model = None
                self.tokenizer = None
        else:
            logger.info("Neural models disabled. Using static monitoring.")
        self.load_hardware_data()
        logger.info("AIHardwareManager initialized.")

    def load_hardware_data(self):
        """Load static hardware optimization data from JSON file."""
        kb_path = os.path.join(self.data_dir, 'hardware_kb.json')
        try:
            with open(kb_path, 'r') as f:
                self.kb = json.load(f)
                logger.info("Loaded hardware knowledge base.")
        except Exception as e:
            logger.error(f"Failed to load hardware_kb.json: {e}. Using default.")
            self.kb = {
                "optimization": {
                    "cpu_high": "Reduce CPU-intensive tasks; prioritize lightweight models.",
                    "memory_low": "Increase virtual memory or upgrade RAM; close unused applications."
                },
                "upgrades": {
                    "gpu": "Consider a GPU with at least 12GB VRAM for better neural performance.",
                    "ram": "Upgrade to 16GB+ RAM for smoother multitasking."
                }
            }

    def monitor_resources(self):
        """Monitor current hardware resource usage.

        Returns:
            dict: {"success": bool, "stats": dict or "message": str}
        """
        try:
            stats = {
                "cpu_usage": psutil.cpu_percent(interval=1),
                "memory_usage": psutil.virtual_memory().percent,
                "memory_total_gb": psutil.virtual_memory().total / (1024 ** 3),
                "gpu_available": torch.cuda.is_available(),
                "gpu_memory_total_gb": torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) if torch.cuda.is_available() else 0,
                "gpu_memory_used_gb": torch.cuda.memory_allocated(0) / (1024 ** 3) if torch.cuda.is_available() else 0
            }
            logger.info(f"Hardware stats: {stats}")
            if self.memory:
                self.memory.save_to_long_term_memory("hardware_stats", stats)
            return {"success": True, "stats": stats}
        except Exception as e:
            logger.error(f"Error monitoring resources: {e}")
            return {"success": False, "message": f"Error: {e}"}

    def advise(self, request):
        """Advise on hardware optimization or upgrades.

        Args:
            request (str): Query (e.g., "Optimize performance", "Upgrade advice").

        Returns:
            dict: {"success": bool, "advice": str or "message": str}
        """
        try:
            stats = self.monitor_resources()["stats"] if self.monitor_resources()["success"] else {}
            if self.neural_capable and self.model and self.tokenizer:
                prompt = f"As a hardware manager, advise based on this request and stats: {request}\nStats: {json.dumps(stats)}"
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                with torch.no_grad():
                    outputs = self.model.generate(inputs["input_ids"], max_length=500, num_beams=4)
                advice = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                if "optimize" in request.lower():
                    advice = []
                    if stats.get("cpu_usage", 0) > 80:
                        advice.append(self.kb["optimization"]["cpu_high"])
                    if stats.get("memory_usage", 0) > 90 or stats.get("memory_total_gb", 0) < 8:
                        advice.append(self.kb["optimization"]["memory_low"])
                    advice = "\n".join(advice) or "No optimization needed based on current stats."
                elif "upgrade" in request.lower():
                    advice = f"{self.kb['upgrades']['gpu']}\n{self.kb['upgrades']['ram']}"
                else:
                    advice = "Specify 'optimize' or 'upgrade' for targeted advice."
            logger.info(f"Provided advice for '{request}': {advice}")
            if self.memory:
                self.memory.save_to_long_term_memory(f"hardware_{hash(request)}", {"request": request, "advice": advice})
            return {"success": True, "advice": advice}
        except Exception as e:
            logger.error(f"Error advising on '{request}': {e}")
            return {"success": False, "message": f"Error: {e}"}

if __name__ == "__main__":
    from ai_assistant.core.memory_manager import MemoryManager
    memory = MemoryManager()
    hw_manager = AIHardwareManager(memory)
    print(hw_manager.monitor_resources())
    print(hw_manager.advise("Optimize performance"))
    print(hw_manager.advise("Upgrade advice"))