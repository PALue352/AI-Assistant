# ai_assistant/core/task_models/business_ai.py
import logging
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import cadquery as cq
from cadquery import exporters

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BusinessAI:
    def __init__(self, memory_manager=None, model_name="Qwen/Qwen-1_8B"):  # Smaller model for RX 580
        logger.info(f"BusinessAI initializing with model: {model_name}...")
        self.memory = memory_manager
        self.resource_dir = os.path.join(os.getenv('PROGRAMDATA', '{userdocs}\\AI_Assistant'), 'Resources')
        self.data_dir = os.path.join(os.path.dirname(__file__), '..', 'training_data')
        os.makedirs(self.resource_dir, exist_ok=True)
        self.templates = self.load_templates()
        self.neural_capable = torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= 8 * 1024**3
        if self.neural_capable:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model.to("cuda" if torch.cuda.is_available() else "cpu")
                logger.info(f"Neural model {model_name} initialized with 4-bit quantization.")
            except Exception as e:
                logger.error(f"Failed to initialize neural model: {e}. Falling back to templates.")
                self.model = None
                self.tokenizer = None
        else:
            logger.info("Neural models disabled due to hardware constraints. Using templates.")
            self.model = None
            self.tokenizer = None
        logger.info("BusinessAI initialized.")

    def load_templates(self):
        """Load patent templates from JSON file."""
        template_path = os.path.join(self.data_dir, 'patent_templates.json')
        try:
            with open(template_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load patent_templates.json: {e}. Using default.")
            return {
                "abstract": "Abstract: {description}",
                "claims": ["1. {description}", "2. The device of claim 1 with {additional_detail}"]
            }

    def generate_patent_text(self, invention_description):
        """Generate patent text using neural model or templates.

        Args:
            invention_description (str): Description of the invention.

        Returns:
            dict: {"success": bool, "text": str or "message": str}
        """
        try:
            if self.neural_capable and self.model and self.tokenizer:
                prompt = (
                    "Draft a patent abstract and claims based on the following invention description:\n"
                    f"{invention_description}\n\n"
                    "Abstract:\n[Insert abstract here]\n\n"
                    "Claims:\n1. [Insert claim 1]\n2. [Insert claim 2]\n"
                )
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                with torch.no_grad():
                    outputs = self.model.generate(inputs["input_ids"], max_length=500, num_beams=4)
                patent_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                abstract = self.templates["abstract"].format(description=invention_description)
                claims = "\n".join(c.format(description=invention_description, additional_detail="enhanced functionality") 
                                   for c in self.templates["claims"])
                patent_text = f"{abstract}\n\nClaims:\n{claims}"
            logger.info("Patent text generated successfully.")
            if self.memory:
                self.memory.save_to_long_term_memory(f"patent_{hash(invention_description)}", {"text": patent_text})
            return {"success": True, "text": patent_text}
        except Exception as e:
            logger.error(f"Error generating patent text: {e}")
            return {"success": False, "message": f"Error: {e}"}

    def generate_patent_image(self, object_description, output_path):
        """Generate a simple CAD-like image for a patent.

        Args:
            object_description (str): Description of the object (e.g., "cube with hole").
            output_path (str): Path to save the image.

        Returns:
            dict: {"success": bool, "path": str or "message": str}
        """
        try:
            if "cube" in object_description.lower():
                shape = cq.Workplane("XY").box(10, 10, 10)
                if "hole" in object_description.lower():
                    shape = shape.faces(">Z").workplane().hole(2)
            else:
                shape = cq.Workplane("XY").box(5, 5, 5)
            exporters.export(shape, output_path, exporters.ExportTypes.SVG)
            logger.info(f"Patent image generated at {output_path}")
            return {"success": True, "path": output_path}
        except Exception as e:
            logger.error(f"Error generating patent image: {e}")
            return {"success": False, "message": f"Error: {e}"}

    def create_patent(self, invention_description, image_output_path="patent_image.svg"):
        """Create a complete patent draft with text and image.

        Args:
            invention_description (str): Description of the invention.
            image_output_path (str): Path for the generated image.

        Returns:
            dict: {"success": bool, "text": str, "image_path": str or "message": str}
        """
        text_result = self.generate_patent_text(invention_description)
        if not text_result["success"]:
            return {"success": False, "message": text_result["message"]}
        
        image_result = self.generate_patent_image(invention_description, image_output_path)
        if not image_result["success"]:
            return {"success": False, "message": image_result["message"]}

        return {
            "success": True,
            "text": text_result["text"],
            "image_path": image_result["path"]
        }

if __name__ == "__main__":
    from ai_assistant.core.memory_manager import MemoryManager
    memory = MemoryManager()
    business = BusinessAI(memory)
    result = business.create_patent("A cube with a cylindrical hole through the center for fluid flow.")
    print(result["text"] if result["success"] else result["message"])
    print(f"Image saved at: {result['image_path']}" if result["success"] else "")