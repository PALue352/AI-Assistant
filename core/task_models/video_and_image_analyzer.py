# ai_assistant/core/task_models/video_and_image_analyzer.py
import logging
import os
import json
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoAndImageAnalyzer:
    def __init__(self, memory_manager=None, model_name="Qwen/Qwen-1_8B"):
        logger.info("VideoAndImageAnalyzer initializing...")
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
                logger.error(f"Failed to initialize neural model: {e}. Using static analysis.")
                self.model = None
                self.tokenizer = None
        else:
            logger.info("Neural models disabled. Using static analysis.")
        self.load_analyzer_data()
        logger.info("VideoAndImageAnalyzer initialized.")

    def load_analyzer_data(self):
        """Load static analyzer data from JSON file."""
        kb_path = os.path.join(self.data_dir, 'analyzer_kb.json')
        try:
            with open(kb_path, 'r') as f:
                self.kb = json.load(f)
                logger.info("Loaded analyzer knowledge base.")
        except Exception as e:
            logger.error(f"Failed to load analyzer_kb.json: {e}. Using default.")
            self.kb = {
                "image_scenes": {
                    "person": "A person is present in the image.",
                    "object": "An object is visible."
                },
                "video_actions": {
                    "movement": "Something is moving in the video."
                }
            }

    def analyze_file(self, file_path):
        """Analyze image or video content in human terms.

        Args:
            file_path (str): Path to the file.

        Returns:
            dict: {"success": bool, "description": str or "message": str}
        """
        try:
            if not os.path.exists(file_path):
                return {"success": False, "message": f"File '{file_path}' not found."}
            
            ext = os.path.splitext(file_path)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                file_type = "image"
                with Image.open(file_path) as img:
                    metadata = {"size": img.size, "format": img.format}
            else:
                file_type = "video"  # Placeholder—video unsupported yet
                metadata = {"type": "video"}

            if self.neural_capable and self.model and self.tokenizer:
                prompt = f"Analyze this {file_type} and describe it in human terms: Metadata={json.dumps(metadata)}"
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                with torch.no_grad():
                    outputs = self.model.generate(inputs["input_ids"], max_length=500, num_beams=4)
                description = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                if file_type == "image":
                    description = self.kb["image_scenes"]["person"]  # Simplified default
                else:
                    description = self.kb["video_actions"]["movement"]
            
            logger.info(f"Analyzed '{file_path}': {description}")
            if self.memory:
                self.memory.save_to_long_term_memory(f"analyze_{hash(file_path)}", {"file_path": file_path, "description": description})
            return {"success": True, "description": description}
        except Exception as e:
            logger.error(f"Error analyzing '{file_path}': {e}")
            return {"success": False, "message": f"Error: {e}"}

if __name__ == "__main__":
    from ai_assistant.core.memory_manager import MemoryManager
    memory = MemoryManager()
    analyzer = VideoAndImageAnalyzer(memory)
    print(analyzer.analyze_file("test_image.jpg"))