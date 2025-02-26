# ai_assistant/core/task_models/ai_anti_virus_firewall.py
import logging
import os
import json
import hashlib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AIAntiVirusFirewall:
    def __init__(self, memory_manager=None, model_name="Qwen/Qwen-1_8B"):
        logger.info("AIAntiVirusFirewall initializing...")
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
                logger.error(f"Failed to initialize neural model: {e}. Using static strategies.")
                self.model = None
                self.tokenizer = None
        else:
            logger.info("Neural models disabled. Using static strategies.")
        self.load_av_data()
        logger.info("AIAntiVirusFirewall initialized.")

    def load_av_data(self):
        """Load static anti-virus data from JSON file."""
        kb_path = os.path.join(self.data_dir, 'av_kb.json')
        try:
            with open(kb_path, 'r') as f:
                self.kb = json.load(f)
                logger.info("Loaded AV knowledge base.")
        except Exception as e:
            logger.error(f"Failed to load av_kb.json: {e}. Using default.")
            self.kb = {
                "threat_signatures": {
                    "malicious_hash": "eicar_test_file_hash",  # Example: EICAR test file hash
                    "suspicious_strings": ["malware", "virus", "exploit"]
                },
                "actions": {
                    "quarantine": "Move file to quarantine folder and alert user.",
                    "delete": "Delete file and log action."
                }
            }

    def scan_file(self, file_path):
        """Scan a file for threats.

        Args:
            file_path (str): Path to the file to scan.

        Returns:
            dict: {"success": bool, "threat": str, "advice": str or "message": str}
        """
        try:
            if not os.path.exists(file_path):
                return {"success": False, "message": f"File '{file_path}' not found."}
            
            # Calculate file hash
            with open(file_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            
            # Heuristic checks
            threat = "None"
            advice = "File appears safe."
            with open(file_path, 'r', errors='ignore') as f:
                content = f.read().lower()
                for sig in self.kb["threat_signatures"]["suspicious_strings"]:
                    if sig in content:
                        threat = "Potential Malware"
                        advice = self.kb["actions"]["quarantine"]
                        break
            
            if file_hash == self.kb["threat_signatures"]["malicious_hash"]:
                threat = "Known Malicious File"
                advice = self.kb["actions"]["delete"]
            
            if self.neural_capable and self.model and self.tokenizer:
                prompt = f"Analyze this file hash and content snippet for threats: Hash={file_hash}, Content={content[:100]}"
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                with torch.no_grad():
                    outputs = self.model.generate(inputs["input_ids"], max_length=500, num_beams=4)
                neural_advice = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                advice += f"\nNeural Analysis: {neural_advice}"

            logger.info(f"Scanned '{file_path}': Threat={threat}, Advice={advice}")
            if self.memory:
                self.memory.save_to_long_term_memory(f"av_{hash(file_path)}", {"file_path": file_path, "threat": threat, "advice": advice})
            return {"success": True, "threat": threat, "advice": advice}
        except Exception as e:
            logger.error(f"Error scanning '{file_path}': {e}")
            return {"success": False, "message": f"Error: {e}"}

if __name__ == "__main__":
    from ai_assistant.core.memory_manager import MemoryManager
    memory = MemoryManager()
    av_ai = AIAntiVirusFirewall(memory)
    print(av_ai.scan_file("test_file.txt"))