import logging
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageProcessingAI:
    def __init__(self):
        logger.info("ImageProcessingAI initializing with model: nlpconnect/vit-gpt2-image-captioning...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Allow GPU (ROCm/OpenCL) detection, fall back to CPU (Reverted from CPU-only override)
        self.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.model.to(self.device)
        logger.info(f"ImageProcessingAI initialized on {self.device}.")

    def caption_image(self, image):
        """Generate a caption for an image using the model.

        Args:
            image: PIL Image or numpy array representing the image.

        Returns:
            dict: {"success": bool, "caption": str or "message": str}
        """
        try:
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs)
            caption = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Generated caption for image: {caption[:50]}...")
            return {"success": True, "caption": caption}
        except Exception as e:
            logger.error(f"Error captioning image: {e}")
            return {"success": False, "message": str(e)}

if __name__ == "__main__":
    from PIL import Image
    image = Image.open("example.jpg")  # Replace with actual image path
    image_processor = ImageProcessingAI()
    result = image_processor.caption_image(image)
    print(result)