# gui_interface.py (v1.0.6) - Corrected Overseer Initialization
import gradio as gr
import asyncio
import logging
import sys
import os
from datetime import datetime
import speech_recognition as sr
from gtts import gTTS
import pyttsx3
import shutil
import glob
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ai_assistant.core.overseer import Overseer
from ai_assistant.core.network_manager import NetworkManager

# Get the root logger (configured in overseer.py)
logger = logging.getLogger(__name__)

class GUIInterface:
    def __init__(self):
        logger.info("Starting GUIInterface initialization")
        self.overseer = None  # Initialize overseer to None
        self.overseer_running = False  # Flag for overseer status
        try:
            # self.overseer = Overseer(gui_interface=self) # Moved to start_overseer
            self.network_manager = NetworkManager()
            self.resource_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model_cache')
            self.model_dir = self.resource_dir
            self.ollama_dir = os.path.expanduser("~/.ollama/models")
            os.makedirs(self.resource_dir, exist_ok=True)
            self.has_microphone = self.check_microphone()
            self.use_voice = self.has_microphone
            self.thinking_time_limit = 18  # Default 18s
            self.iface = self.create_interface()
            logger.info("GUIInterface initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize GUIInterface: {e}", exc_info=True)
            raise RuntimeError(f"Initialization failed: {e}")

    def check_microphone(self):
        try:
            with sr.Microphone() as source:
                logger.info("Microphone detected and tested successfully.")
                return True
        except (OSError, sr.WaitTimeoutError, sr.UnknownValueError, sr.RequestError, ImportError) as e:
            logger.info(f"No microphone detected: {str(e)}. Voice features disabled.")
            return False

    async def process_input(self, user_input):
        logger.info(f"Processing user input: {user_input[:50]}...")
        if not self.overseer or not self.overseer.running:  # Check if Overseer exists and is running
            return {"type": "text", "content": "AI Assistant is not running."}
        try:
            response = await self.overseer.process_request_async(user_input, timeout=self.thinking_time_limit)
            if not isinstance(response, dict) or "type" not in response or "content" not in response:
                logger.error(f"Invalid response format: {response}")
                return {"type": "error", "content": "Invalid response from AI engine"}
            if self.use_voice and response["type"] == "text":
                await self.text_to_speech(response["content"])  # Await the TTS
            return response
        except Exception as e:
            logger.error(f"Error processing input '{user_input[:50]}...': {e}", exc_info=True)
            return {"type": "error", "content": f"Error: {e}"}

    def sync_process_input(self, user_input, chat_history):
        logger.info(f"Submit button clicked with input: {user_input[:50]}...")
        # Use asyncio.run to run the async function in a new event loop.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        future = asyncio.ensure_future(self.process_input(user_input), loop=loop) #Added future
        response = loop.run_until_complete(future) # Run the future object

        if response["type"] == "text":
            chat_history.append((user_input, response["content"]))
        elif response["type"] == "web":
            chat_history.append((user_input, f"<pre>{response['content']}</pre>"))
        elif response["type"] == "error":
            chat_history.append((user_input, f"Error: {response['content']}"))
        else:
            chat_history.append((user_input, f"```json\n{json.dumps(response['content'], indent=2)}\n```"))
        return "", chat_history

    def stop_thinking(self): #This function is no longer async
        logger.info("Stop Thinking button clicked")
        self.overseer.ai_engine.stop_generation()  # NOT async
        return "Stopped AI thinking"

    def collect_feedback(self, user_input, ai_response, user_rating, user_comment):
        logger.info("Feedback button clicked")
        feedback = {'rating': user_rating, 'comment': user_comment}
        try:
            self.overseer.collect_feedback(user_input, ai_response, feedback)
            return "Feedback submitted. Thank you!"
        except Exception as e:
            logger.error(f"Error collecting feedback: {e}", exc_info=True)
            return f"Error submitting feedback: {e}"

    def analyze_file(self, file_data):
        logger.info("Analyze File button clicked")
        if not file_data:
            return "No files uploaded."
        try:
            results = []
            for file in file_data if isinstance(file_data, list) else [file_data]:
                with open(file.name, 'r', encoding='utf-8') as f:
                    content = f.read()
                if file.name.endswith('.py'):
                    explanation = self.overseer.sub_ais["coder"].explain_code(content)
                    results.append(f"Analysis of {os.path.basename(file.name)}:\n{explanation}")
                elif file.name.endswith('.tex'):
                    results.append(f"Analysis of {os.path.basename(file.name)}: LaTeX file detected (analysis TBD)")
                else:
                    results.append(f"File type not supported: {os.path.basename(file.name)}")
            return "\n\n".join(results)
        except Exception as e:
            logger.error(f"Error analyzing files: {e}", exc_info=True)
            return f"Error analyzing files: {e}"

    def set_user_preferences(self, preferences):
        logger.info("Apply Preferences button clicked")
        try:
            self.overseer.set_user_preferences(preferences)
            return "Preferences applied successfully."
        except Exception as e:
            logger.error(f"Error setting preferences: {e}", exc_info=True)
            return f"Error setting preferences: {e}"

    def drop_resources(self, files):
        logger.info("Resource Drop uploaded")
        if not files:
            return "No resources dropped."
        try:
            results = []
            for file in files:
                dest_path = os.path.join(self.resource_dir, os.path.basename(file.name))
                shutil.copy(file.name, dest_path)
                results.append(f"Copied {os.path.basename(file.name)} to {self.resource_dir}")
            return "\n\n".join(results)
        except Exception as e:
            logger.error(f"Error dropping resources: {e}", exc_info=True)
            return f"Error: {e}"

    def list_models(self):
        logger.info("Refresh Model List button clicked")
        try:
            model_files = glob.glob(os.path.join(self.model_dir, "**"), recursive=True)
            models = [os.path.basename(f) for f in model_files if os.path.isdir(f) or f.endswith(('.bin', '.pt', '.gguf'))]
            resource_models = "\n".join(models) if models else "No models found in model_cache."

            ollama_files = glob.glob(os.path.join(self.ollama_dir, "**"), recursive=True)
            ollama_models = [os.path.basename(f) for f in ollama_files if os.path.isdir(f) or f.endswith(('.bin', '.gguf'))]
            ollama_result = "\n".join(ollama_models) if ollama_models else "No models found in Ollama folder."

            return f"Model Cache ({self.model_dir}):\n{resource_models}\n\nOllama Models ({self.ollama_dir}):\n{ollama_result}"
        except Exception as e:
            logger.error(f"Error listing models: {e}", exc_info=True)
            return f"Error: {e}"

    def remove_model(self, model_name):
        logger.info("Remove Model button clicked")
        try:
            model_path = os.path.join(self.model_dir, model_name)
            if os.path.isdir(model_path) and model_name not in ["Qwen-Qwen-7B", "Qwen-7B", "qwen-1_8b"]:
                shutil.rmtree(model_path)
                logger.info(f"Removed model directory: {model_path}")
                return f"Model {model_name} removed successfully."
            elif os.path.isfile(model_path) and model_name not in ["Qwen-Qwen-7B", "Qwen-7B", "qwen-1_8b"]:
                os.remove(model_path)
                logger.info(f"Removed model file: {model_path}")
                return f"Model {model_name} removed successfully."
            else:
                return f"Cannot remove critical model {model_name} or model not found."
        except Exception as e:
            logger.error(f"Error removing model {model_name}: {e}", exc_info=True)
            return f"Error removing model: {e}"

    def set_model_storage(self, new_path):
        logger.info("Set Storage Path button clicked")
        try:
            if not os.path.exists(new_path):
                os.makedirs(new_path, exist_ok=True)
            self.model_dir = new_path
            return f"Model storage set to {new_path}"
        except Exception as e:
            logger.error(f"Error setting model storage: {e}", exc_info=True)
            return f"Error: {e}"

    def change_model(self, model_name):
        logger.info("Apply Model button clicked")
        try:
            result = self.overseer.ai_engine.change_model(model_name)
            return result["content"] if isinstance(result, dict) else result
        except Exception as e:
            logger.error(f"Error changing model: {e}", exc_info=True)
            return f"Error: {e}"

    def set_thinking_time_limit(self, time_limit):
        logger.info("Set Thinking Time button clicked")
        try:
            if isinstance(time_limit, str):
                if time_limit.endswith('h'):
                    seconds = int(time_limit[:-1]) * 3600
                elif time_limit.endswith('d'):
                    seconds = int(time_limit[:-1]) * 86400
                else:
                    seconds = int(time_limit)
            else:
                seconds = int(time_limit)
            self.thinking_time_limit = max(1, seconds)
            logger.info(f"Thinking time limit set to {self.thinking_time_limit} seconds")
            return f"Thinking time limit set to {self.thinking_time_limit} seconds"
        except ValueError as e:
            logger.error(f"Invalid time limit format: {e}", exc_info=True)
            return f"Error: Invalid time limit format (use seconds, 'xh' for hours, or 'xd' for days)"

    def enable_ollama(self, enable):
        logger.info("Ollama toggle changed")
        try:
            if enable:
                self.overseer.ai_engine.use_ollama = True
                import ollama
                self.overseer.ai_engine.client = ollama.Client()
                self.overseer.ai_engine.client.pull(self.overseer.ai_engine.model_name)
                logger.info(f"Ollama enabled with model: {self.overseer.ai_engine.model_name}")
                return "Ollama enabled successfully."
            else:
                self.overseer.ai_engine.use_ollama = False
                logger.info("Ollama disabled, using Transformers.")
                return "Ollama disabled, using Transformers."
        except Exception as e:
            logger.error(f"Failed to enable/disable Ollama: {e}", exc_info=True)
            self.overseer.ai_engine.use_ollama = False
            return f"Failed to enable/disable Ollama: {e}"

    def voice_to_text(self):
        logger.info("Start Listening button clicked")
        if not self.use_voice or not self.has_microphone:
            logger.info("Voice input requested but disabled or no microphone available")
            return ""
        try:
            with sr.Microphone() as source:
                logger.info("Listening for voice input...")
                audio = sr.Recognizer().listen(source, timeout=5, phrase_time_limit=10)
                text = sr.Recognizer().recognize_google(audio)
                logger.info(f"Recognized voice input: {text}")
                return text
        except sr.WaitTimeoutError:
            logger.warning("No speech detected within timeout.")
            return ""
        except sr.UnknownValueError:
            logger.warning("Could not understand audio.")
            return ""
        except sr.RequestError as e:
            logger.error(f"Speech recognition error: {e}", exc_info=True)
            return ""

    async def text_to_speech(self, text):
        if not self.use_voice or not self.has_microphone:
            return
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            logger.info("Played text-to-speech response.")
        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}", exc_info=True)
            try:
                tts = gTTS(text=text, lang='en')
                audio_file = "response.mp3"
                tts.save(audio_file)
                os.system(f"start {audio_file}")  # Use 'start' for Windows
                logger.info("Played text-to-speech via gTTS.")
            except Exception as e2:
                logger.error(f"Error in gTTS fallback: {e2}", exc_info=True)



    def toggle_voice(self, enable):
        logger.info("Voice toggle changed")
        self.use_voice = enable and self.has_microphone
        status = "Voice enabled" if self.use_voice else "Voice disabled (no mic or manually off)"
        logger.info(status)
        return status

    def voice_chat(self):
        text = self.voice_to_text()
        if text:
            response = asyncio.run_coroutine_threadsafe(self.process_input(text), self.loop).result()
            if response["type"] == "text":
                return response["content"]
            return f"Error: {response['content']}" if response["type"] == "error" else str(response)
        return "No voice input detected."

    def start_overseer(self, model_name=None):
        """Starts the Overseer with the specified model."""
        if self.overseer_running:
            logger.info("Overseer already running.  Restarting.")
            self.stop_overseer()

        # Use the provided model_name, or default to "qwen-1_8b" if None
        selected_model = model_name if model_name else "qwen-1_8b"
        logger.info(f"Starting Overseer with model: {selected_model}")

        try:
            self.overseer = Overseer(gui_interface=self, model_name=selected_model)
            self.overseer_running = True
            return "AI Assistant started successfully." #Status update
        except Exception as e:
            logger.error(f"Failed to start Overseer: {e}", exc_info=True)
            return f"Error: {e}"


    def stop_overseer(self):
        """Stops the Overseer."""
        if self.overseer and self.overseer_running:
            logger.info("Stopping Overseer...")
            try:
                self.overseer.stop()  # Ensure stop is called
            except Exception as e:
                logger.error(f"Error stopping Overseer: {e}", exc_info=True)
            finally:
                self.overseer = None  # Remove reference
                self.overseer_running = False # Update running flag
                logger.info("Overseer stopped.")
        else:
            logger.info("Overseer not running.")


    def restart_overseer(self, model_name):
        """Restarts the Overseer with a new model."""
        self.stop_overseer()  # Stop if running
        return self.start_overseer(model_name)  # Restart with selected model

    def get_available_models(self):
        """Returns a list of available models in the model_cache directory."""
        model_files = glob.glob(os.path.join(self.model_dir, "**"), recursive=True)
        # Filter to include only directories that *seem* like model directories
        models = [os.path.basename(f) for f in model_files if os.path.isdir(f) and any(file.endswith((".bin",".pt",".gguf",".safetensors")) for file in os.listdir(f))]
        return models


    def create_interface(self):
        with gr.Blocks(title="AI Assistant", theme=gr.themes.Soft()) as iface:
            with gr.Row():
                gr.Markdown("# AI Assistant\nA customizable, local AI assistant with internet access.")
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Chat History", scale=1)
                with gr.Row():
                    user_input = gr.Textbox(placeholder="Enter your question or command...", label="Your Input", lines=2, scale=4)
                    submit_button = gr.Button("Submit", scale=1, size="sm")
                    stop_button = gr.Button("Stop Thinking", scale=1, size="sm")
                clear = gr.Button("Clear Chat", scale=1)
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group("Feedback"):
                        feedback_rating = gr.Slider(1, 5, step=1, label="Rate the Response", value=3)
                        feedback_comment = gr.Textbox(label="Feedback Comments", placeholder="Optional comments...")
                        feedback_button = gr.Button("Submit Feedback")
                        feedback_status = gr.Textbox(label="Feedback Status", interactive=False)
                    with gr.Group("File Analysis"):
                        file_upload = gr.File(label="Upload Files for Analysis", file_count="multiple", file_types=['.py', '.tex'])
                        analyze_button = gr.Button("Analyze File")
                        file_analysis_result = gr.Textbox(label="File Analysis Result", interactive=False, lines=3)
                with gr.Column(scale=1):
                    with gr.Group("Resources"):
                        resource_drop = gr.File(label="Drag and Drop Resource Files/Folders", file_count="multiple")
                        resource_status = gr.Textbox(label="Resource Drop Status", interactive=False)
            with gr.Accordion("Advanced Settings", open=False):
                with gr.Column():
                    with gr.Row():
                        # Model selection dropdown
                        model_dropdown = gr.Dropdown(label="Select Model", choices=self.get_available_models(), value="qwen-1_8b")
                        restart_button = gr.Button("Restart with Selected Model")
                    with gr.Row():
                        # change_model_input = gr.Textbox(placeholder="e.g., Qwen/Qwen-1_8B", label="Change Model") #Removed
                        # change_model_button = gr.Button("Apply Model") #Removed
                        model_status = gr.Textbox(label="Model Status", interactive=False)
                    with gr.Row():
                        preferences_input = gr.Textbox(placeholder="e.g., 'prefer short answers'", label="Customize Behavior")
                        apply_prefs_button = gr.Button("Apply Preferences")
                        prefs_status = gr.Textbox(label="Preferences Status", interactive=False)
                    with gr.Row():
                        model_list = gr.Textbox(label="Current Models", value=self.list_models(), interactive=False)
                        list_models_button = gr.Button("Refresh Model List")
                    with gr.Row():
                        remove_model_input = gr.Textbox(placeholder="e.g., qwen-1_8b", label="Remove Model")
                        remove_model_button = gr.Button("Remove Model")
                        remove_status = gr.Textbox(label="Remove Status", interactive=False)
                    with gr.Row():
                        model_storage_input = gr.Textbox(placeholder=f"Current: {self.model_dir}", label="Set Model Storage Path")
                        set_storage_button = gr.Button("Set Storage Path")
                        storage_status = gr.Textbox(label="Storage Status", interactive=False)
                    with gr.Row():
                        thinking_time_input = gr.Textbox(placeholder=f"Current: {self.thinking_time_limit}s (e.g., 18, '1h', '1d')", label="Set Thinking Time Limit")
                        set_thinking_time_button = gr.Button("Set Thinking Time")
                        thinking_time_status = gr.Textbox(label="Thinking Time Status", interactive=False)
                    with gr.Row():
                        ollama_toggle = gr.Checkbox(label="Enable Ollama (requires internet)", value=False)
                        ollama_status = gr.Textbox(label="Ollama Status", interactive=False)
                    with gr.Row():
                        voice_toggle = gr.Checkbox(label="Enable Voice Recognition", value=self.has_microphone, interactive=self.has_microphone)
                        voice_status = gr.Textbox(label="Voice Status", value="Enabled" if self.has_microphone else "Disabled (No Mic)", interactive=False)
                        voice_btn = gr.Button("Start Listening")

            submit_button.click(fn=self.sync_process_input, inputs=[user_input, chatbot], outputs=[user_input, chatbot])
            stop_button.click(fn=self.stop_thinking, inputs=None, outputs=gr.Textbox(label="Stop Status", interactive=False))
            clear.click(lambda: [], None, chatbot, queue=False)
            feedback_button.click(fn=self.collect_feedback,
                                inputs=[user_input, chatbot, feedback_rating, feedback_comment],
                                outputs=feedback_status)
            analyze_button.click(fn=self.analyze_file, inputs=file_upload, outputs=file_analysis_result)
            resource_drop.upload(fn=self.drop_resources, inputs=resource_drop, outputs=resource_status)
            # change_model_button.click(fn=self.change_model, inputs=change_model_input, outputs=model_status) #Removed
            apply_prefs_button.click(fn=self.set_user_preferences, inputs=preferences_input, outputs=prefs_status)
            list_models_button.click(fn=self.list_models, inputs=None, outputs=model_list)
            remove_model_button.click(fn=self.remove_model, inputs=remove_model_input, outputs=remove_status)
            set_storage_button.click(fn=self.set_model_storage, inputs=model_storage_input, outputs=storage_status)
            set_thinking_time_button.click(fn=self.set_thinking_time_limit, inputs=thinking_time_input, outputs=thinking_time_status)
            ollama_toggle.change(fn=self.enable_ollama, inputs=ollama_toggle, outputs=ollama_status)
            voice_toggle.change(fn=self.toggle_voice, inputs=voice_toggle, outputs=voice_status)
            voice_btn.click(fn=self.voice_chat, outputs=gr.Textbox(label="Voice Response", interactive=False))
            #Wire up the model selection dropdown
            restart_button.click(fn=self.restart_overseer, inputs=model_dropdown, outputs=model_status)

        return iface

    def run(self):
        logger.info("Launching GUIInterface")
        try:
            #Remove all async calls.
            self.start_overseer() #Start on launch
            self.iface.launch(inbrowser=True)

        except Exception as e:
            logger.error(f"Failed to launch interface: {e}", exc_info=True)
            raise RuntimeError(f"Launch failed: {e}")
        finally:
            if self.overseer:
                self.overseer.stop()
            logger.info("GUIInterface shut down")

    def start_overseer(self, model_name = "qwen-1_8b"):
        '''Start the overseer with the selected model'''
        logger.info(f"Starting Overseer with model: {model_name}")
        try:
            self.overseer = Overseer(gui_interface=self, model_name=model_name)
            self.overseer_running = True
            return "AI Assistant started successfully." #Status update
        except Exception as e:
            logger.error(f"Failed to start the Overseer: {e}", exc_info = True)
            return f"Error Occurred: {e}"

    def stop_overseer(self):
        '''Stop the overseer'''
        logger.info("Stopping Overseer")
        if self.overseer and self.overseer_running:
            self.overseer.stop()
            self.overseer_running = False
            self.overseer = None
        else:
            logger.info("Overseer was not running.")


    def restart_overseer(self, model_name):
        """Restarts the Overseer with a new model."""
        self.stop_overseer()  # Stop if running
        return self.start_overseer(model_name)  # Restart with selected model