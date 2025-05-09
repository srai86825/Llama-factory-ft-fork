from huggingface_hub import HfApi, create_repo
import os
import shutil

# This approach doesn't require loading the full model
# It simply uploads the LoRA weights directly

# Path to your LoRA adapter
lora_path = "/workspace/saves/Qwen2_5-32b-instruct-qa-agent/lora/sft/"

# Define your repository
repo_id = "srai86825/qwen-vl-tool-assistant-lora"

# Create a model card
model_card = """
# Qwen2.5-VL-32B Tool Assistant with LoRA fine-tuning

This is a LoRA adapter for the Qwen2.5-VL-32B model, fine-tuned for tool-use with visual input.

## Usage

```python
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel
import torch
from PIL import Image

# Load the model
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-32B-Instruct")
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-VL-32B-Instruct", 
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(
    base_model, 
    "srai86825/qwen-vl-tool-assistant-lora"
)

# Use the model
image = Image.open("your_image.jpg")
text = "What is in this image?"

inputs = processor(text=text, images=image, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
result = processor.decode(outputs[0], skip_special_tokens=True)
print(result)
```

## Training Details
- Base model: Qwen/Qwen2.5-VL-32B-Instruct
- Fine-tuning method: LoRA with rank 8
- Target modules: all
- Training data: Custom tool-use dataset
"""

# Save the model card to the adapter directory
with open(os.path.join(lora_path, "README.md"), "w") as f:
    f.write(model_card)

# Initialize the Hugging Face API
api = HfApi()

# First, create the repository
print(f"Creating repository {repo_id}...")
create_repo(
    repo_id=repo_id,
    repo_type="model",
    exist_ok=True  # Won't error if the repo already exists
)

# Then upload the adapter files to Hugging Face Hub
print("Uploading to Hugging Face Hub...")
api.upload_folder(
    folder_path=lora_path,
    repo_id=repo_id,
    repo_type="model"
)
print("Done!")