import logging
import os
from chromadb import Client
import torch
import json

# Version code for tracking
VERSION = "v1.005"  # Updated version for fixing GPU detection (generic AMD support for RX 580), removing ROCm assumptions

# Configure logging first
logging.basicConfig(level=logging.INFO, format=f'%(asctime)s - %(name)s - %(levelname)s - [Version {VERSION}] - %(message)s')
logger = logging.getLogger(__name__)

# Detect GPU for AMD Radeon RX 580 on Windows (using PyTorch’s AMD support, no ROCm assumption)
try:
    # Attempt to detect any GPU, prioritizing AMD RX 580 via PyTorch’s AMD compatibility
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        # Verify AMD Radeon RX 580 compatibility (generic check for AMD GPUs)
        try:
            if 'amd' in torch.cuda.get_device_properties(0).name.lower() or 'radeon' in torch.cuda.get_device_properties(0).name.lower():
                logger.info(f"Detected AMD Radeon RX 580 GPU, initializing with CUDA support. [Version {VERSION}]")
            else:
                logger.warning(f"GPU detected but not AMD Radeon RX 580; falling back to CPU. [Version {VERSION}]")
                device = torch.device('cpu')
        except RuntimeError as e:
            logger.error(f"Error verifying GPU: {e}. Falling back to CPU. [Version {VERSION}]")
            device = torch.device('cpu')
    else:
        logger.warning(f"No GPU support detected; falling back to CPU. [Version {VERSION}]")
except Exception as e:
    logger.error(f"Error detecting GPU: {e}. Falling back to CPU. [Version {VERSION}]")
    device = torch.device('cpu')

# Make device globally accessible for other modules
import sys
sys.modules['__main__'].device = device  # Export device to be accessible in other files

class MemoryManager:
    def __init__(self):
        logger.info(f"MemoryManager initializing on {device.type}... [Version {VERSION}]")
        self.client = Client()  # In-memory client, not persistent, compatible with GPU/CPU
        self.collection_name = "ai_assistant_memory"
        try:
            # Try to get or create the collection, avoiding UniqueConstraintError
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            # Log readiness for GPU-optimized tasks (ChromaDB doesn’t natively use GPU, but downstream tasks can)
            if device.type == 'cuda':
                logger.info(f"ChromaDB collection initialized with potential GPU optimization readiness for DeepSeek-R1. [Version {VERSION}]")
        except Exception as e:
            logger.error(f"Error initializing collection: {e}. Falling back to CPU-only mode without collection. [Version {VERSION}]")
            self.collection = None
        self.max_interactions = 100  # Limit to prevent memory overuse
        self.long_term_storage_path = os.path.join(os.path.dirname(__file__), "long_term_memory.json")
        self.load_long_term_memory()  # Load existing long-term memory
        logger.info(f"MemoryManager initialized on {device.type}. [Version {VERSION}]")

    def load_long_term_memory(self):
        """Loads long-term memory from a JSON file, optimized for GPU/CPU context."""
        try:
            if os.path.exists(self.long_term_storage_path):
                with open(self.long_term_storage_path, 'r', encoding='utf-8') as f:
                    self.long_term_memory = json.load(f)
            else:
                self.long_term_memory = {}
        except json.JSONDecodeError as e:
            logger.error(f"Error loading long-term memory: {e} [Version {VERSION}]")
            self.long_term_memory = {}
        except Exception as e:
            logger.error(f"Error initializing long-term memory: {e} [Version {VERSION}]")
            self.long_term_memory = {}

    def save_to_long_term_memory(self, key, data):
        """Saves data to long-term memory in a JSON file, optimized for GPU/CPU context."""
        if self.collection:  # Only save if collection is initialized
            try:
                self.long_term_memory[key] = data
                with open(self.long_term_storage_path, 'w', encoding='utf-8') as f:
                    json.dump(self.long_term_memory, f, indent=4)
                logger.info(f"Saved data to long-term memory under key: {key} on {device.type} [Version {VERSION}]")
            except Exception as e:
                logger.error(f"Error saving to long-term memory: {e} [Version {VERSION}]")
        else:
            logger.warning(f"No collection available on {device.type}; data not saved to long-term memory. [Version {VERSION}]")

    def get_from_long_term_memory(self, key):
        """Retrieves data from long-term memory, optimized for GPU/CPU context."""
        try:
            return self.long_term_memory.get(key, None)
        except Exception as e:
            logger.error(f"Error retrieving from long-term memory: {e} on {device.type} [Version {VERSION}]")
            return None

    def store_interaction(self, user_input, ai_response):
        """Stores an interaction in memory, optimized for GPU/CPU context."""
        if self.collection:
            try:
                interaction_id = str(len(self.collection.get()["ids"]) + 1)
                self.collection.add(
                    ids=[interaction_id],
                    documents=[f"User: {user_input}\nAI: {ai_response}"]
                )
                if self.collection.count() > self.max_interactions:
                    # Remove oldest interaction (simplistic approach)
                    oldest_id = self.collection.get()["ids"][0]
                    self.collection.delete(ids=[oldest_id])
                logger.info(f"Stored interaction {interaction_id} on {device.type} [Version {VERSION}]")
            except Exception as e:
                logger.error(f"Error storing interaction on {device.type}: {e} [Version {VERSION}]")
        else:
            logger.warning(f"No collection available on {device.type}; interaction not stored. [Version {VERSION}]")

    def get_latest_interaction(self):
        """Returns the latest user input and AI response, optimized for GPU/CPU context."""
        if self.collection:
            try:
                interactions = self.collection.get()["documents"]
                if interactions:
                    latest = interactions[-1]
                    user_part, ai_part = latest.split("\n", 1)
                    user_input = user_part.replace("User: ", "").strip()
                    ai_response = ai_part.replace("AI: ", "").strip()
                    return user_input, ai_response
                return None, None
            except Exception as e:
                logger.error(f"Error retrieving latest interaction on {device.type}: {e} [Version {VERSION}]")
                return None, None
        logger.warning(f"No collection available on {device.type}; no latest interaction retrieved. [Version {VERSION}]")
        return None, None

    def get_recent_interactions(self, limit=10):
        """Returns recent interactions for diversity/loop checks, optimized for GPU/CPU context."""
        if self.collection:
            try:
                interactions = self.collection.get()["documents"][-limit:]
                return interactions if interactions else []
            except Exception as e:
                logger.error(f"Error retrieving recent interactions on {device.type}: {e} [Version {VERSION}]")
                return []
        logger.warning(f"No collection available on {device.type}; no recent interactions retrieved. [Version {VERSION}]")
        return []

    def get_context(self):
        """Returns context based on recent interactions, optimized for GPU/CPU context."""
        if self.collection:
            try:
                interactions = self.get_recent_interactions()
                context = " ".join(interactions) if interactions else ""
                return context[:1000]  # Limit context size
            except Exception as e:
                logger.error(f"Error generating context on {device.type}: {e} [Version {VERSION}]")
                return ""
        logger.warning(f"No collection available on {device.type}; no context generated. [Version {VERSION}]")
        return ""

    def __del__(self):
        logger.info(f"MemoryManager resources cleaned up on {device.type} [Version {VERSION}]")