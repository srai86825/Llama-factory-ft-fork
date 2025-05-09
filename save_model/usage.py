from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load the processor and base model
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-32B-Instruct")
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-VL-32B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# Load your LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "srai86825/qwen-vl-tool-assistant-lora"
)

# Now the model can be used with your fine-tuning