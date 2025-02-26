import logging
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ollama import Client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KnowledgeBaseAI:
    def __init__(self, ai_engine=None, memory_manager=None):
        logger.info("KnowledgeBaseAI initializing...")
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
                logger.info(f"Loaded {model_name} model for KnowledgeBaseAI on {self.device} with 4-bit quantization using bitsandbytes-0.45.2 (GPU attempt with ROCm/OpenCL, matching early logs).")
            except Exception as e:
                logger.error(f"Failed to initialize neural model with GPU: {e}. Falling back to CPU/static knowledge base.")
                self.model = None
                self.tokenizer = None
                self.neural_capable = False
        if not self.neural_capable:
            logger.info("No model specified or neural incapable. Using static knowledge base on CPU.")
        self.load_knowledge_base()
        logger.info(f"KnowledgeBaseAI initialized on {self.device}.")

    def load_knowledge_base(self):
        """Load knowledge base from JSON file or memory."""
        kb_path = os.path.join(self.data_dir, 'knowledge_base.json')
        try:
            with open(kb_path, 'r') as f:
                self.knowledge_base = json.load(f)
                logger.info("Loaded knowledge base from file.")
        except FileNotFoundError:
            logger.warning(f"Knowledge base not found at {kb_path}. Using default data.")
            self.knowledge_base = {"fact": "The Earth has one moon.", "definition": "Gravity is the force pulling objects toward each other."}
        except Exception as e:
            logger.error(f"Failed to load knowledge_base.json: {e}. Using default data.")
            self.knowledge_base = {"fact": "The Earth has one moon.", "definition": "Gravity is the force pulling objects toward each other."}
        if self.memory:
            cached = self.memory.get_from_long_term_memory("knowledge_cache")
            if cached:
                self.knowledge_base.update(cached.get("data", {}))
                logger.info("Updated knowledge base from long-term memory.")

    def retrieve_knowledge(self, query):
        """Retrieve knowledge based on a query using LLM or static templates.

        Args:
            query (str): Knowledge query (e.g., "How many moons does Earth have?").

        Returns:
            dict: {"success": bool, "knowledge": str or "message": str}
        """
        try:
            if self.neural_capable and self.model and self.tokenizer:
                prompt = f"Retrieve knowledge for: {query}\nUse this knowledge: {json.dumps(self.knowledge_base)}"
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model.generate(inputs["input_ids"], max_length=500, num_beams=4)
                knowledge = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                if "moons" in query.lower() and "earth" in query.lower():
                    knowledge = self.knowledge_base["fact"]
                elif "gravity" in query.lower():
                    knowledge = self.knowledge_base["definition"]
                else:
                    knowledge = "Knowledge not found; specify a known fact or definition."
            logger.info(f"Retrieved knowledge for '{query}': {knowledge[:50]}...")
            if self.memory:
                self.memory.save_to_long_term_memory(f"knowledge_{hash(query)}", {"query": query, "knowledge": knowledge})
            return {"success": True, "knowledge": knowledge}
        except Exception as e:
            logger.error(f"Error retrieving knowledge for '{query}': {e}")
            return {"success": False, "message": str(e)}

if __name__ == "__main__":
    from ai_assistant.core.memory_manager import MemoryManager
    from ai_assistant.core.ai_engine import AIEngine
    memory = MemoryManager()
    ai_engine = AIEngine(model_name="deepseek-r1:1.5b")
    kb_ai = KnowledgeBaseAI(ai_engine, memory)
    print(kb_ai.retrieve_knowledge("How many moons does Earth have?"))
    print(kb_ai.retrieve_knowledge("What is gravity?"))