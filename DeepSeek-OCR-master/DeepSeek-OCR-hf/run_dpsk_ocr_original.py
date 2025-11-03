"""
A library by Hugging Face that provides ready-to-use AI models (like LLMs, OCR, etc.).
AutoModel → Automatically loads a pre-trained model (e.g., text, OCR, vision, etc.).

AutoTokenizer → Loads the corresponding tokenizer for the model (handles text input/output).

"""

from transformers import AutoModel, AutoTokenizer 
import torch #PyTorch, used for deep learning computations and GPU acceleration.
import os


os.environ["CUDA_VISIBLE_DEVICES"] = '0' #this is to specify which GPU to use (if multiple GPUs are available).


model_name = 'deepseek-ai/DeepSeek-OCR'


tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)  #tAllows the model’s custom loading logic
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, use_safetensors=True,device_map='cuda:0',low_cpu_mem_usage=True,torch_dtype=torch.bfloat16)
"""
Uses a safer format for model weights (faster & safer)
device_map='cuda:0'	Runs the model on GPU 0
low_cpu_mem_usage=True	Reduces CPU memory usage while loading the model
torch_dtype=torch.bfloat16	Uses bfloat16 precision — faster than float32 and uses less memory, ideal for modern GPUs

"""

model = model.eval()
"""
Sets the model to evaluation mode.

This disables training-related features (like dropout) for stable and consistent inference (prediction).
"""



# prompt = "<image>\n<|grounding|>Convert the document to markdown. "
prompt = "<image>\nFree OCR. "
image_file = 'sample.png'
output_path = './output'
torch.set_default_device('cuda:0')


# infer(self, tokenizer, prompt='', image_file='', output_path = ' ', base_size = 1024, image_size = 640, crop_mode = True, test_compress = False, save_results = False):

# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 640, image_size = 640, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False

# Gundam: base_size = 1024, image_size = 640, crop_mode = True
with torch.cuda.device('cuda:0'):
    res = model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path = output_path, base_size = 1024, image_size = 640, crop_mode=True, save_results = True, test_compress = True)
