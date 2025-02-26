import logging
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Force CPU mode for RX 580 on Windows
torch.cuda.is_available = lambda: False
device = torch.device('cpu')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinancialAdvisorAI:
    def __init__(self, ai_engine=None, memory_manager=None, model_name=None):
        logger.info("FinancialAdvisorAI initializing...")
        self.ai_engine = ai_engine
        self.memory = memory_manager
        self.resource_dir = os.path.join(os.getenv('PROGRAMDATA', '{userdocs}\\AI_Assistant'), 'Resources')
        self.data_dir = os.path.join(os.path.dirname(__file__), '..', 'training_data')
        os.makedirs(self.resource_dir, exist_ok=True)
        self.neural_capable = torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= 8 * 1024**3
        self.model = None
        self.tokenizer = None
        model_name = model_name or (self.ai_engine.model_name if self.ai_engine else "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        if self.neural_capable:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model.to(device)  # Force CPU for RX 580
                logger.info(f"Neural model {model_name} initialized with 4-bit quantization on CPU.")
            except Exception as e:
                logger.error(f"Failed to initialize neural model: {e}. Using static advice.")
                self.model = None
                self.tokenizer = None
        else:
            logger.info("Neural models disabled. Using static advice.")
        self.load_financial_data()
        logger.info("FinancialAdvisorAI initialized.")

    def load_financial_data(self):
        """Load static financial data from JSON file."""
        kb_path = os.path.join(self.data_dir, 'financial_kb.json')
        try:
            with open(kb_path, 'r') as f:
                self.kb = json.load(f)
                logger.info("Loaded financial knowledge base.")
        except Exception as e:
            logger.error(f"Failed to load financial_kb.json: {e}. Using default.")
            self.kb = {
                "budgeting": {
                    "rule": "50/30/20 rule: 50% needs, 30% wants, 20% savings/debt repayment.",
                    "steps": ["Calculate monthly income.", "Allocate 50% to needs (housing, food).", "30% to wants (entertainment).", "20% to savings/debt."]
                },
                "investing": {
                    "basic_advice": "Diversify investments; consider low-cost index funds.",
                    "risk": "Higher risk may yield higher returns but increases loss potential."
                }
            }

    def advise(self, request):
        """Provide financial advice based on user request.

        Args:
            request (str): Financial query (e.g., "How to budget $2000 monthly?", "Should I invest in stocks?").

        Returns:
            dict: {"success": bool, "advice": str or "message": str}
        """
        try:
            if self.neural_capable and self.model and self.tokenizer:
                prompt = f"As a financial advisor, provide detailed advice for: {request}"
                inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = self.model.generate(inputs["input_ids"], max_length=500, num_beams=4)
                advice = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                if "budget" in request.lower():
                    income_str = ''.join(filter(str.isdigit, request)) or "0"
                    income = float(income_str) if income_str else 0
                    advice = (
                        f"{self.kb['budgeting']['rule']}\n"
                        f"For ${income} monthly:\n"
                        f"- Needs: ${income * 0.5}\n"
                        f"- Wants: ${income * 0.3}\n"
                        f"- Savings/Debt: ${income * 0.2}\n"
                        f"Steps: {', '.join(self.kb['budgeting']['steps'])}"
                    )
                elif "invest" in request.lower():
                    advice = f"{self.kb['investing']['basic_advice']} Note: {self.kb['investing']['risk']}"
                else:
                    advice = "Please specify budgeting or investing advice."
            logger.info(f"Provided advice for '{request}': {advice}")
            if self.memory:
                self.memory.save_to_long_term_memory(f"finance_{hash(request)}", {"request": request, "advice": advice})
            return {"success": True, "advice": advice}
        except Exception as e:
            logger.error(f"Error advising on '{request}': {e}")
            return {"success": False, "message": f"Error: {e}"}

if __name__ == "__main__":
    from ai_assistant.core.memory_manager import MemoryManager
    from ai_assistant.core.ai_engine import AIEngine
    memory = MemoryManager()
    ai_engine = AIEngine(model_name="deepseek-r1:1.5b")
    advisor = FinancialAdvisorAI(ai_engine, memory)
    print(advisor.advise("How to budget $2000 monthly?"))
    print(advisor.advise("Should I invest in stocks?"))