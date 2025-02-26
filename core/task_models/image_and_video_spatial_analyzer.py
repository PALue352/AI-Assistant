import logging
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ollama import Client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageAndVideoSpatialAnalyzer:
    def __init__(self, ai_engine=None, memory_manager=None):
        logger.info("ImageAndVideoSpatialAnalyzer initializing...")
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
                logger.info(f"Loaded {model_name} model for ImageAndVideoSpatialAnalyzer on {self.device} with 4-bit quantization using bitsandbytes-0.45.2 (GPU attempt with ROCm/OpenCL, matching early logs).")
            except Exception as e:
                logger.error(f"Failed to initialize neural model with GPU: {e}. Falling back to CPU/static analysis.")
                self.model = None
                self.tokenizer = None
                self.neural_capable = False
        if not self.neural_capable:
            logger.info("Neural models disabled. Using static analysis on CPU.")
        self.load_spatial_knowledge()
        logger.info(f"ImageAndVideoSpatialAnalyzer initialized on {self.device}.")

    def load_spatial_knowledge(self):
        """Load spatial analysis knowledge base from JSON file or memory."""
        kb_path = os.path.join(self.data_dir, 'spatial_kb.json')
        try:
            with open(kb_path, 'r') as f:
                self.spatial_data = json.load(f)
                logger.info("Loaded spatial knowledge base from file.")
        except FileNotFoundError:
            logger.warning(f"Spatial knowledge base not found at {kb_path}. Using default data.")
            self.spatial_data = {"object_detection": "Identify objects by size, position, and motion.", "scene_analysis": "Analyze layout, depth, and perspective."}
        except Exception as e:
            logger.error(f"Failed to load spatial_kb.json: {e}. Using default data.")
            self.spatial_data = {"object_detection": "Identify objects by size, position, and motion.", "scene_analysis": "Analyze layout, depth, and perspective."}
        if self.memory:
            cached = self.memory.get_from_long_term_memory("spatial_cache")
            if cached:
                self.spatial_data.update(cached.get("data", {}))
                logger.info("Updated spatial data from long-term memory.")

    def analyze_spatial_data(self, data_description):
        """Analyze spatial data in images or videos using LLM or static templates.

        Args:
            data_description (str): Description of spatial data (e.g., "Object at top-left corner of video frame").

        Returns:
            dict: {"success": bool, "analysis": str or "message": str}
        """
        try:
            if self.neural_capable and self.model and self.tokenizer:
                prompt = f"Analyze spatial data: {data_description}\nUse this knowledge: {json.dumps(self.spatial_data)}"
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model.generate(inputs["input_ids"], max_length=500, num_beams=4)
                analysis = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                if "object" in data_description.lower() and "corner" in data_description.lower():
                    analysis = self.spatial_data["object_detection"]
                elif "layout" in data_description.lower() or "depth" in data_description.lower():
                    analysis = self.spatial_data["scene_analysis"]
                else:
                    analysis = "Cannot analyze; specify object position or scene characteristics."
            logger.info(f"Analyzed spatial data '{data_description}': {analysis[:50]}...")
            if self.memory:
                self.memory.save_to_long_term_memory(f"spatial_{hash(data_description)}", {"description": data_description, "analysis": analysis})
            return {"success": True, "analysis": analysis}
        except Exception as e:
            logger.error(f"Error analyzing spatial data '{data_description}': {e}")
            return {"success": False, "message": str(e)}

if __name__ == "__main__":
    from ai_assistant.core.memory_manager import MemoryManager
    from ai_assistant.core.ai_engine import AIEngine
    memory = MemoryManager()
    ai_engine = AIEngine(model_name="deepseek-r1:1.5b")
    spatial_analyzer = ImageAndVideoSpatialAnalyzer(ai_engine, memory)
    print(spatial_analyzer.analyze_spatial_data("Object at top-left corner of video frame"))
    print(spatial_analyzer.analyze_spatial_data("Analyze layout and depth in image"))