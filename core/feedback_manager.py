# feedback_manager.py (v1.0.2)
import logging
import os
import json
import time  # Import the time module
#Needed to take this import out to prevent circular import.  The device is determined by ai_engine from now on
#from .overseer import device  # Import device from overseer

# Get the root logger (configured in overseer.py)
logger = logging.getLogger(__name__)

class FeedbackManager:
    def __init__(self):
        logger.info("FeedbackManager initializing...")
        self.feedback_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback.json")
        self.feedback_data = self.load_feedback()
        #logger.info(f"Device set to use {device}") #Removed
        logger.info("FeedbackManager initialized.")

    def load_feedback(self):
        try:
            if os.path.exists(self.feedback_file):
                with open(self.feedback_file, 'r') as f:
                    return json.load(f)
            else:
                logger.info(f"Feedback file {self.feedback_file} not found, starting with empty feedback.")
                return {}  # Initialize as an empty dictionary
        except Exception as e:
            logger.error(f"Error loading feedback: {e}. Starting with empty feedback.", exc_info=True)
            return {} # Return empty dict on failure


    def collect_feedback(self, query, response, user_feedback):
        timestamp = int(time.time())  # Use time.time() for POSIX timestamp
        self.feedback_data[timestamp] = {
            'query': query,
            'response': response,
            'user_feedback': user_feedback
        }
        self.save_feedback()

    def save_feedback(self):
        try:
            with open(self.feedback_file, 'w') as f:
                json.dump(self.feedback_data, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving feedback: {e}", exc_info=True)

    def get_feedback_for_training(self):
        # Placeholder: In a real system, this would filter and format feedback for training
        return self.feedback_data

    def __del__(self):
        self.save_feedback()
        logger.info("FeedbackManager resources cleaned up.")