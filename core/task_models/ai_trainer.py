# ai_assistant/core/task_models/ai_trainer.py
import logging
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
from typing import Optional, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AITrainer:
    def __init__(self, overseer: 'Overseer', model_name: str = "Qwen/Qwen-1_8B"):
        logger.info("AITrainer initializing...")
        self.overseer = overseer  # Reference to Overseer instance
        self.data_dir = os.path.join(os.path.dirname(__file__), '..', 'training_data')
        self.resource_dir = os.path.join(os.getenv('PROGRAMDATA', '{userdocs}\\AI_Assistant'), 'Resources')
        os.makedirs(self.output_dir, exist_ok=True)
        self.neural_capable = self.overseer.neural_capable
        self.model_name = model_name
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        if self.neural_capable:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model.to("cuda" if torch.cuda.is_available() else "cpu")
                logger.info(f"Neural model {model_name} initialized with 4-bit quantization.")
            except Exception as e:
                logger.error(f"Failed to initialize neural model: {e}. Training disabled.")
                self.model = None
                self.tokenizer = None
        else:
            logger.info("Neural models disabled due to hardware constraints.")
        self.load_training_data()
        logger.info("AITrainer initialized.")

    def load_training_data(self) -> None:
        """Load initial training data from ethical_training_data.json."""
        ethical_data_path = os.path.join(self.data_dir, 'ethical_training_data.json')
        try:
            with open(ethical_data_path, 'r') as f:
                self.training_data = json.load(f).get("ethical_examples", [])
            logger.info("Loaded ethical training data.")
        except Exception as e:
            logger.error(f"Failed to load ethical_training_data.json: {e}. Using empty dataset.")
            self.training_data = []

    def extract_structured_data(self, response: str, category: str) -> Dict[str, Any]:
        """Extract structured data from training responses to populate JSON files."""
        try:
            if category == "medical" and "symptoms" in response.lower():
                symptoms = response.split("Possible causes:")[0].strip().split(":")[0].capitalize()
                causes = response.split("Possible causes:")[1].split("Advice:")[0].strip().split(", ")
                advice = response.split("Advice:")[1].strip()
                return {"symptoms": {symptoms: {"causes": causes, "advice": advice}}}
            elif category == "knowledge" and "Fact" in response:
                fact_key = response.split("Fact:")[1].split(":")[0].strip().lower().replace(" ", "_")
                fact_value = response.split("Fact:")[1].split(":")[1].strip()
                return {"facts": {fact_key: fact_value}}
            elif category == "cognitive" and "Reasoning" in response.lower():
                reasoning_type = response.split(":")[0].strip().lower()
                explanation = response.split(":")[1].strip()
                return {"reasoning": {reasoning_type: explanation}}
            return {}
        except Exception as e:
            logger.error(f"Error extracting structured data: {e}")
            return {}

    def tokenize_function(self, examples):
        """Tokenize training examples."""
        return self.tokenizer(examples["input"], truncation=True, padding="max_length", max_length=128, return_tensors="pt")

    def train(self, category: str = "general", epochs: int = 1) -> Dict[str, Any]:
        """Fine-tune the model and populate JSON files."""
        if not self.neural_capable or not self.model or not self.tokenizer:
            return {"success": False, "message": "Neural training not available on this hardware."}

        try:
            # Prepare dataset
            dataset = Dataset.from_list(self.training_data)
            tokenized_dataset = dataset.map(self.tokenize_function, batched=True)
            tokenized_dataset = tokenized_dataset.remove_columns(["input"])  # Remove non-tensor columns

            # Training arguments
            training_args = TrainingArguments(
                output_dir=os.path.join(self.output_dir, f"{self.model_name}_{category}"),
                num_train_epochs=epochs,
                per_device_train_batch_size=4,
                save_steps=500,
                save_total_limit=2,
                logging_dir=os.path.join(self.output_dir, 'logs'),
                logging_steps=10,
            )

            # Define Trainer with default loss (e.g., cross-entropy for causal LM)
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset,
            )

            # Train the model
            logger.info(f"Starting training for category: {category}")
            trainer.train()
            self.model.save_pretrained(os.path.join(self.output_dir, f"{self.model_name}_{category}"))
            self.tokenizer.save_pretrained(os.path.join(self.output_dir, f"{self.model_name}_{category}"))
            logger.info(f"Training completed for {category}. Model saved.")

            # Generate and extract structured data
            enhanced_data = {}
            for example in self.training_data[:5]:
                input_text = example["input"]
                prompt = f"Provide a detailed response to: {input_text}"
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                with torch.no_grad():
                    outputs = self.model.generate(inputs["input_ids"], max_length=300, num_beams=4)
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                structured_data = self.extract_structured_data(response, category)
                if structured_data:
                    enhanced_data.update(structured_data)
                    if hasattr(self.overseer, 'ai_watcher') and self.overseer.ai_watcher:
                        self.overseer.ai_watcher.monitor(f"Extracted {category} data", json.dumps(structured_data))

            # Update JSON files
            if enhanced_data:
                json_file_map = {
                    "medical": "medical_kb.json",
                    "knowledge": "knowledge_base.json",
                    "cognitive": "cognitive_kb.json"
                }
                json_file = json_file_map.get(category, "general_training_data.json")
                json_path = os.path.join(self.data_dir, json_file)
                try:
                    existing_data = json.load(open(json_path, 'r')) if os.path.exists(json_path) else {}
                    existing_data.update(enhanced_data)
                    with open(json_path, 'w') as f:
                        json.dump(existing_data, f, indent=4)
                    logger.info(f"Updated {json_file} with training-derived data.")
                except Exception as e:
                    logger.error(f"Failed to update {json_file}: {e}")

            return {"success": True, "message": f"Training completed for {category}. JSON files updated."}
        except Exception as e:
            logger.error(f"Error during training: {e}")
            return {"success": False, "message": f"Error during training: {e}"}

if __name__ == "__main__":
    from ai_assistant.core.overseer import Overseer
    overseer = Overseer()
    trainer = AITrainer(overseer)
    result = trainer.train("cognitive")
    print(result["message"])