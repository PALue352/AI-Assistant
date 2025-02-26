import logging
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Force CPU mode for RX 580 on Windows (AMD GPU not supporting CUDA)
torch.cuda.is_available = lambda: False
device = torch.device('cpu')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PhysicsAI:
    def __init__(self, memory_manager=None, ai_engine=None, model_name=None):
        logger.info("PhysicsAI initializing...")
        self.memory = memory_manager  # Added to accept memory_manager parameter (Issue: TypeError in overseer.py due to missing memory_manager parameter; Fix: Updated __init__ to accept memory_manager)
        self.ai_engine = ai_engine
        self.resource_dir = os.path.join(os.getenv('PROGRAMDATA', '{userdocs}\\AI_Assistant'), 'Resources')
        self.data_dir = os.path.join(os.path.dirname(__file__), '..', 'training_data')
        os.makedirs(self.resource_dir, exist_ok=True)
        self.neural_capable = torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= 8 * 1024**3
        self.model = None
        self.tokenizer = None
        model_name = model_name or (self.ai_engine.model_name if self.ai_engine else "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        if self.neural_capable:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, load_in_4bit=True)  # Uses bitsandbytes for 4-bit quantization on CPU
                self.model.to(device)  # Force CPU for RX 580
                logger.info(f"Loaded {model_name} model for PhysicsAI on CPU with 4-bit quantization using bitsandbytes-0.45.2.")
            except Exception as e:
                logger.error(f"Failed to initialize neural model: {e}. Using static strategies or no model specified.")
                self.model = None
                self.tokenizer = None
        else:
            logger.info("Neural models disabled or no model specified. Using static strategies or no model specified on CPU.")
        self.load_physics_data()
        logger.info("PhysicsAI initialized on CPU.")

    def load_physics_data(self):
        """Load physics knowledge base from JSON file or memory."""
        kb_path = os.path.join(self.data_dir, 'physics_kb.json')
        try:
            with open(kb_path, 'r') as f:
                self.physics_data = json.load(f)
                logger.info("Loaded physics knowledge base from file.")
        except FileNotFoundError:
            logger.warning(f"Physics knowledge base not found at {kb_path}. Using default physics data.")
            self.physics_data = {"gravity": "9.81 m/s²", "velocity": "v = u + at", "force": "F = ma"}
        except Exception as e:
            logger.error(f"Failed to load physics_kb.json: {e}. Using default physics data.")
            self.physics_data = {"gravity": "9.81 m/s²", "velocity": "v = u + at", "force": "F = ma"}
        if self.memory:
            cached = self.memory.get_from_long_term_memory("physics_cache")
            if cached:
                self.physics_data.update(cached.get("data", {}))
                logger.info("Updated physics data from long-term memory.")

    def solve_physics_problem(self, problem):
        """Solve a physics problem using LLM or static templates.

        Args:
            problem (str): Physics problem description (e.g., "Calculate force given mass=10kg and acceleration=2m/s²").

        Returns:
            dict: {"success": bool, "solution": str or "message": str}
        """
        try:
            if self.neural_capable and self.model and self.tokenizer:
                prompt = f"Solve the physics problem: {problem}\nUse this knowledge: {json.dumps(self.physics_data)}"
                inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = self.model.generate(inputs["input_ids"], max_length=500, num_beams=4)
                solution = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                if "force" in problem.lower() and "mass" in problem.lower() and "acceleration" in problem.lower():
                    mass = float(problem.split("mass=")[1].split("kg")[0]) if "kg" in problem else 0
                    accel = float(problem.split("acceleration=")[1].split("m/s²")[0]) if "m/s²" in problem else 0
                    solution = f"F = {mass} * {accel} = {mass * accel} N"
                elif "velocity" in problem.lower() and "initial velocity" in problem.lower() and "acceleration" in problem.lower() and "time" in problem.lower():
                    u = float(problem.split("initial velocity=")[1].split("m/s")[0]) if "m/s" in problem else 0
                    a = float(problem.split("acceleration=")[1].split("m/s²")[0]) if "m/s²" in problem else 0
                    t = float(problem.split("time=")[1].split("s")[0]) if "s" in problem else 0
                    solution = f"v = {u} + {a} * {t} = {u + a * t} m/s"
                else:
                    solution = "Cannot solve; specify force, mass, acceleration, velocity, or related physics terms with units."
            logger.info(f"Solved physics problem '{problem}': {solution[:50]}...")
            if self.memory:
                self.memory.save_to_long_term_memory(f"physics_{hash(problem)}", {"problem": problem, "solution": solution})
            return {"success": True, "solution": solution}
        except Exception as e:
            logger.error(f"Error solving physics problem '{problem}': {e}")
            return {"success": False, "message": str(e)}

if __name__ == "__main__":
    from ai_assistant.core.memory_manager import MemoryManager
    from ai_assistant.core.ai_engine import AIEngine
    memory = MemoryManager()
    ai_engine = AIEngine(model_name="deepseek-r1:1.5b")
    physics_ai = PhysicsAI(memory, ai_engine)
    print(physics_ai.solve_physics_problem("Calculate force given mass=10kg and acceleration=2m/s²"))
    print(physics_ai.solve_physics_problem("Find velocity with initial velocity=5m/s, acceleration=3m/s², time=2s"))