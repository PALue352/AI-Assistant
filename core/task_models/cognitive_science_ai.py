import logging
import os
import json
from ..memory_manager import MemoryManager

# Version code for tracking
VERSION = "v1.003"  # Updated version after fixing TypeError in __init__, maintaining CPU-only static reasoning

logging.basicConfig(level=logging.INFO, format=f'%(asctime)s - %(name)s - %(levelname)s - [Version {VERSION}] - %(message)s')
logger = logging.getLogger(__name__)

class CognitiveScienceAI:
    def __init__(self, ai_engine=None, memory_manager=None):
        logger.info(f"CognitiveScienceAI initializing... [Version {VERSION}]")
        self.memory = memory_manager  # Accept memory_manager directly, no default creation
        if not self.memory:
            self.memory = MemoryManager()  # Create default if None
        self.knowledge_base = {}
        self.load_knowledge_base()
        self.load_reasoning_cache()
        logger.info(f"CognitiveScienceAI initialized on cpu. [Version {VERSION}]")

    def load_knowledge_base(self):
        """Load static knowledge base from cognitive_kb.json."""
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'training_data')
        kb_path = os.path.join(data_dir, 'cognitive_kb.json')
        try:
            with open(kb_path, 'r') as f:
                self.knowledge_base = json.load(f)
                logger.info(f"Loaded cognitive knowledge base. [Version {VERSION}]")
        except Exception as e:
            logger.error(f"Failed to load cognitive_kb.json: {e}. Using default. [Version {VERSION}]")
            self.knowledge_base = {
                "reasoning": {"deduction": "Deriving specific conclusions from general premises.", "induction": "Generalizing from specific observations."},
                "consciousness": {"hard_problem": "Explaining how subjective experience arises from physical processes.", "theories": ["Integrated Information Theory", "Global Workspace Theory"]},
                "neural_networks": {"backpropagation": "Adjusting weights based on error gradients.", "layers": ["Input", "Hidden", "Output"]},
                "pdp": {"principle": "Information processing via interconnected nodes in parallel.", "example": "Word recognition via distributed activation patterns."}
            }

    def load_reasoning_cache(self):
        """Load reasoning data from memory cache."""
        logger.info(f"Loading reasoning cache... [Version {VERSION}]")
        try:
            cached_data = self.memory.get_from_long_term_memory("cognitive_cache")
            if cached_data:
                self.reasoning_steps = cached_data.get("steps", {})
                logger.info(f"Loaded reasoning cache from memory. [Version {VERSION}]")
            else:
                self.initialize_default_cache()
                self.memory.save_to_long_term_memory("cognitive_cache", {"steps": self.reasoning_steps})
                logger.info(f"Initialized and saved default cognitive cache. [Version {VERSION}]")
        except Exception as e:
            logger.error(f"Failed to load reasoning cache: {e}. Using default. [Version {VERSION}]")
            self.initialize_default_cache()

    def initialize_default_cache(self):
        """Initialize default reasoning steps."""
        self.reasoning_steps = {
            "Solve x + 2 = 5": {"problem": "Solve x + 2 = 5", "steps": ["Subtract 2 from both sides: x + 2 - 2 = 5 - 2", "Simplify: x = 3"], "solution": "x = 3"}
        }

    def solve_problem(self, problem):
        """Solves a problem using static reasoning.

        Args:
            problem (str): Problem statement (e.g., "Solve x + 2 = 5" or "What is consciousness?").

        Returns:
            dict: {"success": bool, "steps": list, "solution": str or "message": str}
        """
        try:
            if problem in self.reasoning_steps:
                cached = self.reasoning_steps[problem]
                logger.info(f"Retrieved cached solution for '{problem}' [Version {VERSION}]")
                return {"success": True, "steps": cached["steps"], "solution": cached["solution"]}

            if "=" in problem and any(c.isalpha() for c in problem):
                parts = problem.split("=", 1)
                # Simplified symbolic reasoning (remove sympy dependency)
                left, right = parts[0].strip(), parts[1].strip()
                if left.endswith("x") and right.isdigit():  # Simple case: "x + 2 = 5"
                    steps = [f"Start with: {problem}", f"Subtract {left[:-1]} from both sides: x = {int(right) - int(left[:-1])}"]
                    solution = str(int(right) - int(left[:-1]))
                else:
                    raise ValueError("Complex equation not handled in static mode; use neural model or specify simpler format.")
            else:
                if "reasoning" in problem.lower():
                    key = problem.split("reasoning", 1)[1].strip().lower()
                    response = next((v for k, v in self.knowledge_base["reasoning"].items() if k in key), "Reasoning topic not found.")
                elif "consciousness" in problem.lower():
                    response = f"Theories: {', '.join(self.knowledge_base['consciousness']['theories'])}. Hard Problem: {self.knowledge_base['consciousness']['hard_problem']}"
                elif "neural network" in problem.lower():
                    response = f"Layers: {', '.join(self.knowledge_base['neural_networks']['layers'])}. Backpropagation: {self.knowledge_base['neural_networks']['backpropagation']}"
                elif "parallel distributed" in problem.lower() or "pdp" in problem.lower():
                    response = f"Principle: {self.knowledge_base['pdp']['principle']}. Example: {self.knowledge_base['pdp']['example']}"
                else:
                    response = "Question not recognized in static knowledge base."
                steps = ["Identify query type", f"Retrieve from knowledge base: {response}"]
                solution = response

            self.reasoning_steps[problem] = {"problem": problem, "steps": steps, "solution": solution}
            self.memory.save_to_long_term_memory("cognitive_cache", {"steps": self.reasoning_steps})

            logger.info(f"Solved problem '{problem}': {solution} [Version {VERSION}]")
            return {"success": True, "steps": steps, "solution": solution}
        except Exception as e:
            logger.error(f"Error solving problem '{problem}': {e} [Version {VERSION}]")
            return {"success": False, "message": f"Error: {e}"}

if __name__ == "__main__":
    from ai_assistant.core.memory_manager import MemoryManager
    memory = MemoryManager()
    cog_ai = CognitiveScienceAI(memory_manager=memory)
    print(cog_ai.solve_problem("Solve x + 2 = 5"))
    print(cog_ai.solve_problem("What is consciousness?"))