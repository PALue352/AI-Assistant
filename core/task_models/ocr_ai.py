#ocr_ai.py
import logging
import easyocr
import torch  # Add this line

logger = logging.getLogger(__name__)

class OCRAI:
    def __init__(self):
        logger.info("OCRAI initializing...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Allow GPU (ROCm/OpenCL) detection, fall back to CPU (Reverted from CPU-only override)
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        logger.info(f"OCRAI initialized on {self.device}.")


    def extract_text(self, image_path):
        """Extract text from image using EasyOCR.

        Args:
            image_path (str): Path to image.

        Returns:
            dict: {"success": bool, "text": str or "message": str}
        """
        try:
            result = self.reader.readtext(image_path, detail=0)
            text = " ".join(result) if result else ""
            logger.info(f"Extracted text from {image_path}: {text[:50]}...")
            return {"success": True, "text": text}
        except FileNotFoundError:
            message = f"Image not found: {image_path}"
            logger.error(message)
            return {"success": False, "message": message}
        except Exception as e:
            message = f"Error extracting text from {image_path}: {e}"
            logger.error(message)
            return {"success": False, "message": message}