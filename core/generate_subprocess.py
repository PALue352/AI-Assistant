import sys
import pickle
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate(input_file, output_file):
    try:
        with open(input_file, 'rb') as f:
            request, model_name, resource_dir, timeout = pickle.load(f)
        
        model_path = os.path.join(resource_dir, model_name.replace('/', '-'))
        logger.info(f"Loading model from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda:0", trust_remote_code=True, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        logger.info(f"Generating response for request: {request[:50]}...")
        inputs = tokenizer(request, return_tensors="pt").to("cuda:0")
        outputs = model.generate(inputs["input_ids"], max_length=500, num_beams=4, temperature=0.7, do_sample=True)
        response = {"type": "text", "content": tokenizer.decode(outputs[0], skip_special_tokens=True)}

        with open(output_file, 'wb') as f:
            pickle.dump(response, f)
        logger.info("Response generated and saved")
    except Exception as e:
        logger.error(f"Error in subprocess: {e}", exc_info=True)
        with open(output_file, 'wb') as f:
            pickle.dump({"type": "error", "content": str(e)}, f)

if __name__ == "__main__":
    generate(sys.argv[1], sys.argv[2])