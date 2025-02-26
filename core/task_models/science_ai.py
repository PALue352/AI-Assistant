# ai_assistant/core/task_models/science_ai.py
import logging
import sympy
from sympy import symbols, solve, Eq, simplify, N
import numpy as np
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ScienceAI:
    def __init__(self, memory_manager=None):
        logger.info("ScienceAI initializing...")
        self.memory = memory_manager
        self.resource_dir = os.path.join(os.getenv('PROGRAMDATA', '{userdocs}\\AI_Assistant'), 'Resources')
        os.makedirs(self.resource_dir, exist_ok=True)
        self.load_science_cache()
        logger.info("ScienceAI initialized.")

    def load_science_cache(self):
        """Load science data from Overseer's memory cache."""
        if self.memory:
            cached_data = self.memory.load_from_long_term_memory("science_cache")
            if cached_data:
                self.constants = cached_data.get("constants", {})
                logger.info("Loaded science cache from Overseer memory.")
            else:
                self.initialize_default_cache()
                self.memory.save_to_long_term_memory("science_cache", {"constants": self.constants})
                logger.info("Initialized and saved default science cache to Overseer memory.")
        else:
            self.initialize_default_cache()
            logger.info("No MemoryManager provided; using default science cache.")

    def initialize_default_cache(self):
        """Initialize default scientific constants."""
        self.constants = {
            "G": 6.67430e-11,  # Gravitational constant (m³/kg/s²)
            "h": 6.62607015e-34  # Planck constant (J·s)
        }

    def solve_equation(self, equation):
        """Solves a scientific equation.

        Args:
            equation (str): Equation string (e.g., "F = G * m1 * m2 / r**2").

        Returns:
            dict: {"success": bool, "result": str or "message": str}
        """
        try:
            eq_parts = equation.split("=")
            if len(eq_parts) != 2:
                raise ValueError("Equation must have exactly one '=' sign.")
            left, right = sympify(eq_parts[0]), sympify(eq_parts[1])
            eq = Eq(left, right)
            variables = eq.free_symbols
            if len(variables) > 1:
                raise ValueError("Specify variable to solve for (multiple variables detected).")
            var = variables.pop()
            result = solve(eq, var)
            logger.info(f"Solved equation '{equation}': {result}")
            return {"success": True, "result": str([N(r) for r in result])}
        except Exception as e:
            logger.error(f"Error solving equation: {e}")
            return {"success": False, "message": f"Error: {e}"}

    def analyze_data(self, data):
        """Analyzes scientific data (e.g., mean, std dev).

        Args:
            data (list): Numeric data points.

        Returns:
            dict: {"success": bool, "mean": float, "std_dev": float or "message": str}
        """
        try:
            arr = np.array([float(x) for x in data])
            mean = float(np.mean(arr))
            std_dev = float(np.std(arr))
            logger.info(f"Data analyzed: Mean={mean}, StdDev={std_dev}")
            return {"success": True, "mean": mean, "std_dev": std_dev}
        except Exception as e:
            logger.error(f"Error analyzing data: {e}")
            return {"success": False, "message": f"Error: {e}"}

if __name__ == "__main__":
    from ai_assistant.core.memory_manager import MemoryManager
    memory = MemoryManager()
    science = ScienceAI(memory)
    print(science.solve_equation("F = G * m1 * m2 / r**2"))  # Example gravitational force
    print(science.analyze_data([1, 2, 3, 4, 5]))