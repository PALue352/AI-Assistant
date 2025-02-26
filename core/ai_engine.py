# ai_engine.py (v1.0.9)
import logging
import os
import asyncio
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from datetime import datetime

# Get the root logger (configured in overseer.py)
logger = logging.getLogger(__name__)

# Custom Stopping Criteria (for stop_generation)
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids: list[int], request_ids: dict):
        super().__init__()
        self.stop_token_ids = stop_token_ids
        self.request_ids = request_ids  # Dictionary to track requests to stop

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Check if *any* active request needs to be stopped
        for req_id, should_stop in self.request_ids.items():
            if should_stop:
                return True  # Stop immediately if any request is marked to stop

        # Otherwise, check for stop tokens (normal behavior)
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:  # Check the last generated token
                return True
        return False

class AIEngine:
    def __init__(self, model_name="qwen-1_8b", network_manager=None, device=None):
        logger.info("AIEngine initializing...")
        self.context = []
        self.model_name = model_name
        self.network_manager = network_manager
        self.use_ollama = False
        self.use_internet = os.getenv('USE_INTERNET', 'False').lower() == 'true'
        self.model = None
        self.tokenizer = None
        self.resource_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'Resources', 'Models')
        self.custom_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model_cache')
        os.makedirs(self.resource_dir, exist_ok=True)
        os.makedirs(self.custom_model_dir, exist_ok=True)
        self._process = None  # No longer used
        self.stop_tokens = []  # Stop token IDs
        self.request_ids = {}  # Dictionary to track requests to stop {request_id: stop_flag}
        self.gpu_available = False # Add gpu_available attribute

        # Use the passed device, default to CUDA if available, else CPU
        if device:
            self.device = device
        else:
            self.check_device()
        logger.info(f"Device detected: {self.device}")

        self._initialize_transformers()

    def check_neural_capability(self):
        try:
            if self.gpu_available:
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
                self.gpu_available = True
                self.device =  torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
        except:
            self.device =  torch.device('cpu')

    def _initialize_transformers(self):
        try:
            model_path = os.path.join(self.custom_model_dir, self.model_name.replace(":", "-"))
            # First, check if the model exists locally. Only try to download if it's not there.
            if os.path.exists(model_path):
                logger.info(f"Loading {self.model_name} from {model_path}...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                logger.info(f"Loaded model from: {model_path}")

            else:
                logger.info(f"Model {self.model_name} not found locally. Downloading...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto",  # Let transformers handle device placement
                    trust_remote_code=True,
                    torch_dtype=torch.float16  # Use float16 for efficiency
                )
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                # Save the newly downloaded model and tokenizer
                self.model.save_pretrained(model_path)
                self.tokenizer.save_pretrained(model_path)
                logger.info(f"Model downloaded and saved to: {model_path}")

            # Move model to the correct device (this line is important!)
            self.model.to(self.device)
            logger.info(f"Model loaded on {self.device}")

            # Set up stop tokens for generation
            self.stop_tokens = [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]
            if self.stop_tokens[1] is None:  # Handle cases where pad_token_id is not defined
                self.stop_tokens = [self.stop_tokens[0]]


        except Exception as e:
            logger.error(f"Failed to initialize transformers: {e}", exc_info=True)
            self.model = None
            self.tokenizer = None
            return  # Exit early if model loading fails

    async def process_request(self, request, timeout=18):
        if not self.model or not self.tokenizer:
            return {"type": "error", "content": "No AI model available"}

        logger.info(f"Processing request on {self.device}: {request[:50]}...")

        # --- Input Validation ---
        if not request.strip():  # Check for empty or whitespace-only input
            return {"type": "error", "content": "Input cannot be empty."}

        inputs = self.tokenizer(request, return_tensors="pt").to(self.device)

        # Generate a unique request ID
        request_id = os.urandom(16).hex()  # Generates a random 16-byte hex string
        self.request_ids[request_id] = False  # Initially, don't stop

        # Create the stopping criteria with the current request IDs
        stopping_criteria = StoppingCriteriaList([StopOnTokens(self.stop_tokens, self.request_ids)])

        # Wrap the generation in a timeout.  Remove the await
        try:
            outputs = self.model.generate(**inputs, max_new_tokens=512, stopping_criteria=stopping_criteria)
            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = {"type": "text", "content": response_text}
            logger.info(f"Generated response: {response_text[:50]}...") # Log the generated text

        except asyncio.TimeoutError:
            logger.warning(f"Request timed out after {timeout} seconds.")
            response = {"type": "error", "content": "AI response timed out."}
            # No need to call self.stop_generation() here; timeout is automatic

        except Exception as e:
            logger.error(f"Error during generation: {e}", exc_info=True)
            response = {"type": "error", "content": str(e)}


        self.context.append({'role': 'user', 'content': request})
        self.context.append({'role': 'assistant', 'content': response["content"]})

        # Clean up the request ID (important to prevent memory leaks) AFTER saving to memory
        if request_id in self.request_ids:
            del self.request_ids[request_id]
        return response

    def change_model(self, new_model_name):
        # Unload the current model (important to prevent memory issues)
        if self.model:
            del self.model
            self.model = None
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()  # Clear GPU memory
            logger.info(f"Unloaded previous model.")
        try:
            model_path = os.path.join(self.custom_model_dir, new_model_name.replace(":", "-"))
            if not os.path.exists(model_path):
                logger.info(f"Downloading {new_model_name}...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    new_model_name,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
                self.tokenizer = AutoTokenizer.from_pretrained(new_model_name, trust_remote_code=True)
                self.model.save_pretrained(model_path)  # Save the newly downloaded model
                self.tokenizer.save_pretrained(model_path)
            else:
                logger.info(f"Loading {new_model_name} from {model_path}...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

            self.model.to(self.device)
            self.model_name = new_model_name
            self.context = []
            self.stop_tokens = [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]
            if self.stop_tokens[1] is None:
                self.stop_tokens = [self.stop_tokens[0]]
            return {"type": "text", "content": f"Successfully changed to: {new_model_name}"}

        except Exception as e:
            logger.error(f"Failed to change model: {e}", exc_info=True)
            return {"type": "error", "content": str(e)}

    def stop_generation(self):
        # Set the stop flag for *all* active requests
        logger.info("Stopping generation...")
        for request_id in self.request_ids:
            self.request_ids[request_id] = True
        #return  # Add a return statement here #No longer needed


    def __del__(self):
        if self.model:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("AIEngine resources cleaned up")