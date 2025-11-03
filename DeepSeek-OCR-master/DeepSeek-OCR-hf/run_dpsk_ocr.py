from transformers import AutoModel, AutoTokenizer
import torch
import os
import traceback
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model_name = 'deepseek-ai/DeepSeek-OCR'
# print(f"Python version: {sys.version}")
# print(f"PyTorch version: {torch.__version__}")
# print(f"CUDA available: {torch.cuda.is_available()}")
# if torch.cuda.is_available():
#     print(f"CUDA version: {torch.version.cuda}")
#     print(f"GPU: {torch.cuda.get_device_name(0)}")
#     print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
# try:
    # print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # print("✓ Tokenizer loaded")
    # print("\nLoading model (this may take a while)...")
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_safetensors=True,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map='cuda:0'  # Directly load to GPU
)
model = model.eval()
#     print("✓ Model loaded to GPU")
#     print("✓ Model ready on GPU")
# except Exception as e:
#     print(f"\n:x: ERROR: {e}")
#     traceback.print_exc()
#     exit(1)
# For just OCR text extraction
prompt = "<image>\nFree OCR. "
image_file = './sample.png'
os.makedirs('./output', exist_ok=True)
# print("\nRunning OCR inference...")
# Set default tensor type to CUDA to ensure all tensors are created on GPU
torch.set_default_device('cuda:0')
# try:
with torch.cuda.device('cuda:0'):
    res = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=image_file,
        output_path='output',
        base_size=1024,
        image_size=640,
        crop_mode=True,
        save_results=False,
        test_compress=False
    )
    # print("\n=== Extracted Text ===")
    # print(res)
#     with open('output/extracted_text.txt', 'w', encoding='utf-8') as f:
#         f.write(res)
#     print("\n✓ Text saved to output/extracted_text.txt")
# except Exception as e:
#     # print(f"\n:x: Inference ERROR: {e}")
#     # traceback.print_exc()