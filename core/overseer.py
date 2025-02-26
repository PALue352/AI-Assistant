# overseer.py (v1.0.11 - Debugging Initialization, No psutil, Corrected Logging)
import logging
import asyncio
import os
import time
import importlib
from queue import Queue
from threading import Thread
import sys
# import psutil  # Temporarily commented out for debugging, now put back.
from datetime import datetime
from .memory_manager import MemoryManager
from .ai_engine import AIEngine
from .ai_watcher import AIWatcher
from .feedback_manager import FeedbackManager
from .network_manager import NetworkManager
from .ethical_trainer import EthicalTrainer
from .plugin_manager import PluginManager
from .task_models.common_sense_model import CommonSenseModel
from .task_models.coder_ai import CoderAI
from .task_models.image_processing_ai import ImageProcessingAI
from .task_models.ocr_ai import OCRAI
from .task_models.latex_ai import LatexAI
from .task_models.math_model import MathModel
from .task_models.medical_ai import MedicalAI


# --- Centralized Logging Configuration ---
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'ai_assistant.log')  # Single log file

# Configure the root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='a'),  # Append to log file
        logging.StreamHandler()  # Output to console
    ]
)
logger = logging.getLogger(__name__)  # Get the root logger
# --- End Centralized Logging Configuration ---

# Check for transformers import *after* logger is set up
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    logger.info("Transformers imported successfully.")  # Keep this check
except ImportError:
    logger.warning("Transformers not installed. Model loading will fail.")
    AutoModelForCausalLM = None
    AutoTokenizer = None

logger.info("Overseer starting")


class Overseer:
    def __init__(self, gui_interface=None, model_name="qwen-1_8b"): #Added default parameter
        logger.info("Overseer initializing...")
        self.running = False
        self.memory_manager = MemoryManager()
        logger.info("MemoryManager initialized")  # DEBUG
        self.network_manager = NetworkManager()
        logger.info("NetworkManager initialized")  # DEBUG
        # Initialize AIEngine with Qwen 1.8B, passing device. Device now checked in AI engine
        self.ai_engine = AIEngine(model_name=model_name, network_manager=self.network_manager, device=self.check_device()) #Added model name
        logger.info("AIEngine initialized")  # DEBUG
        self.ai_watcher = AIWatcher(self)
        logger.info("AIWatcher initialized")  # DEBUG
        self.feedback_manager = FeedbackManager()
        logger.info("FeedbackManager initialized")  # DEBUG
        self.ethical_trainer = EthicalTrainer()
        logger.info("EthicalTrainer initialized")  # DEBUG
        self.plugin_manager = PluginManager()
        logger.info("PluginManager initialized")  # DEBUG
        self.plugin_manager.set_overseer(self)
        logger.info("PluginManager set to Overseer")
        self.neural_capable = self.ai_engine.check_neural_capability() # Check neural capability from ai_engine.
        logger.info("Neural Capability checked")
        logger.info(f"Device detected: {self.ai_engine.device}. GPU available: {self.ai_engine.gpu_available}")


        # Qwen 1.8B model path (your custom directory)
        self.qwen_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model_cache', 'qwen-1_8b')
        self.qwen_model = None  # No longer needed here.
        self.qwen_tokenizer = None # No longer needed here.


        # Initialize sub-AIs
        logger.info("Initializing sub-AIs...")
        self.common_sense = CommonSenseModel(self)
        logger.info("CommonSenseModel initialized")  # DEBUG
        self.coder = CoderAI(self.memory_manager)
        logger.info("CoderAI initialized")  # DEBUG
        self.image_processor = ImageProcessingAI()
        logger.info("ImageProcessingAI initialized")  # DEBUG
        self.ocr = OCRAI()
        logger.info("OCRAI initialized")  # DEBUG
        self.latex = LatexAI()
        logger.info("LatexAI initialized")  # DEBUG
        self.math_model = MathModel()
        logger.info("MathModel initialized")  # DEBUG
        self.medical = MedicalAI(self)
        logger.info("MedicalAI initialized")  # DEBUG
        logger.info("...Sub-AIs initialized")

        self.gui_interface = gui_interface
        self.task_queue = Queue()
        self.background_thread = Thread(target=self._run_background_tasks, daemon=True)
        self.background_thread.start()
        # asyncio.run(self.initialize())  # REMOVED - No longer needed
        self.initialize() # Call Synchronously
        logger.info(f"Overseer initialized. Neural capability: {self.neural_capable}")

    def check_neural_capability(self):
        try:
            #Import torch here, so we check if available
            import torch
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                if gpu_memory >= 4:  # Qwen 1.8B needs ~3-4GB
                    logger.info(f"GPU detected with {gpu_memory:.2f}GB VRAM. Neural models enabled.")
                    return True
                else:
                    logger.warning(f"GPU memory insufficient ({gpu_memory:.2f}GB < 4GB). Neural models disabled.")
            cpu_cores = os.cpu_count()
            ram = psutil.virtual_memory().total / (1024 ** 3)
            if ram >= 8 and cpu_cores >= 4:  # Suitable for Qwen 1.8B on CPU
                logger.info(f"CPU with {cpu_cores} cores and {ram:.2f}GB RAM. Neural models enabled.")
                return True
            logger.warning(f"Insufficient CPU/RAM ({cpu_cores} cores, {ram:.2f}GB). Neural models disabled.")
            return False
        except Exception as e:
            logger.error(f"Error checking neural capability: {e}", exc_info=True)
            return False

    def check_device(self):
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        except:
            return "cpu"
    # async def initialize(self): # MODIFIED - No longer async
    def initialize(self):
        # await self.network_manager.initialize() # Removed
        self.running = True
        logger.info("Overseer running")

    async def process_request_async(self, user_input, model="qwen-1_8b", timeout=18):
        if not self.running:
            return {"type": "text", "content": "AI Assistant is not running."}
        response = await self.ai_engine.process_request(user_input, timeout=timeout)
        if not isinstance(response, dict) or "type" not in response or "content" not in response:
            logger.error(f"Invalid response format from AI engine: {response}")
            response = {"type": "text", "content": str(response) if response else "No response"}
        self.memory_manager.store_interaction(user_input, response["content"])
        self.ai_watcher.monitor(user_input, response)
        self.feedback_manager.collect_feedback(user_input, response["content"], {})
        return response

    def stop(self):
        self.running = False
        self.ai_watcher.stop()
        self.ai_engine.stop_generation()
        if self.network_manager:
            asyncio.run(self.network_manager.close())
        logger.info("Overseer stopped")

    def get_latest_interaction(self):
        return self.memory_manager.get_latest_interaction()

    def memory(self):
        return self.memory_manager

    def get_recent_interactions(self):
        return self.memory_manager.get_recent_interactions()

    def process_request(self, request, model="qwen-1_8b"):
        logger.info(f"Received request: {request[:50]}...")
        if "background" in request.lower():
            self.task_queue.put(request)
            return "Task added to background processing queue."
        elif "train ethics" in request.lower():
            self.ethical_trainer.train()
            return "Ethical training initiated."
        else:
            return asyncio.run(self.process_request_async(request, model))

    def _run_background_tasks(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        while self.running:
            if not self.task_queue.empty():
                request = self.task_queue.get()
                try:
                    response = loop.run_until_complete(self.process_request_async(request))
                    logger.info(f"Background task: {request[:50]}... -> {response['content'][:50]}...")
                except Exception as e:
                    logger.error(f"Background task error: {e}")
            else:
                time.sleep(1)

    def start(self):
        if self.running:
            print("Welcome to the AI Assistant. Type 'exit' to quit.")
            while self.running:
                user_input = input(">> ")
                if user_input.lower() == 'exit':
                    self.running = False
                    logger.info("Exiting.")
                else:
                    try:
                        response = self.process_request(user_input)
                        if isinstance(response, dict):
                            print(response["content"])
                        else:
                            print(response)
                    except Exception as e:
                        logger.error(f"Error processing request: {e}")
                        print("An error occurred. Check logs.")

    def prioritize_task(self, task, urgency, importance):
        priority_score = urgency * 0.6 + importance * 0.4
        return priority_score

    def manage_task_queue(self, tasks):
        return sorted(tasks, key=lambda t: self.prioritize_task(t, t.get('urgency', 0), t.get('importance', 0)), reverse=True)

    def load_sub_ai(self, sub_ai_name):
        try:
            module = importlib.import_module(f'.task_models.{sub_ai_name}', package='ai_assistant.core')
            sub_ai = getattr(module, sub_ai_name.capitalize())(self)
            setattr(self, sub_ai_name, sub_ai)
            logger.info(f"Loaded sub-AI: {sub_ai_name}")
            return sub_ai
        except ImportError:
            logger.error(f"Unable to load sub-AI {sub_ai_name}")
            return None

    def unload_sub_ai(self, sub_ai_name):
        if hasattr(self, sub_ai_name):
            delattr(self, sub_ai_name)
            logger.info(f"Unloaded sub-AI: {sub_ai_name}")

    def collect_feedback(self, query, response, user_feedback):
        self.feedback_manager.collect_feedback(query, response, user_feedback)
        logger.info(f"Feedback collected for query: {query[:50]}...")

    def analyze_and_update(self):
        feedback_data = self.feedback_manager.get_feedback_for_training()
        if feedback_data:
            logger.info("Initiating model update based on user feedback.")
        else:
            logger.info("No feedback data available for model update.")

    def set_user_preferences(self, preferences):
        try:
            logger.info(f"User preferences set: {preferences}")
        except Exception as e:
            logger.error(f"Error setting preferences: {e}")

if __name__ == "__main__":
    overseer = Overseer()
    overseer.start()