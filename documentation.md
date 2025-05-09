# Fine-tuning Qwen2.5-VL with LLaMA-Factory

This document covers the steps needed to successfully fine-tune the Qwen2.5-VL-32B-Instruct model using LLaMA-Factory with a custom dataset.

## Prerequisites

- LLaMA-Factory installed
- At least 160GB of GPU VRAM for full 32B parameter model
- Sufficient disk space (>40GB recommended) for model weights

## Step 1: Preparing the Dataset

The Qwen2.5-VL model requires a dataset in the ShareGPT format. Here's the correct structure:

```json
[
  {
    "conversations": [
      {"from": "system", "value": "You are a helpful assistant."},
      {"from": "human", "value": "What is shown in this image?<img>https://example.com/sample1.jpg</img>"},
      {"from": "gpt", "value": "The image shows a mountain landscape with a lake and trees in the foreground."}
    ]
  }
]
```

### Key Format Requirements:
- Each item in the array represents a complete multi-turn conversation
- Each conversation has a "conversations" array containing messages
- Each message has "from" and "value" fields
- "from" can be "system", "human", or "gpt"
- For image inputs, use `<img>URL</img>` tags in the text

## Step 2: Creating the Dataset Configuration

Create a proper `dataset_info.json` file in the following location:

```bash
/workspace/Llama-factory-ft-fork/src/llamafactory/data/dataset_info.json
```

With this content:

```json
{
  "qa_tool_dataset": {
    "file_name": "sample_dataset.json",
    "formatting": "sharegpt",
    "messages": "conversations",
    "role_tag": "from",
    "content_tag": "value",
    "user_tag": "human",
    "assistant_tag": "gpt",
    "system_tag": "system"
  }
}
```

Create the same file in your dataset directory:

```bash
/workspace/Llama-factory-ft-fork/data/qa_tool_dataset/dataset_info.json
```

## Step 3: Configuring Training Parameters

Create a training YAML file (`qwen2_5vl_lora_sft.yaml`):

```yaml
### model
model_name_or_path: Qwen/Qwen2.5-VL-32B-Instruct
image_max_pixels: 262144
video_max_pixels: 16384
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

### dataset
dataset: qa_tool_dataset
dataset_dir: /workspace/Llama-factory-ft-fork/data/qa_tool_dataset
template: qwen2_vl
cutoff_len: 2048
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 1  # Reduce for smaller datasets
dataloader_num_workers: 1     # Reduce for smaller datasets

### output
output_dir: /workspace/saves/Qwen2_5-32b-instruct-qa-agent/lora/sft  # Store in /workspace volume
cache_dir: /workspace/cache  # Store model cache in /workspace volume
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: none

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null
```

### Important YAML Settings:
- Setting `cache_dir: /workspace/cache` moves the model weights to persistent storage
- `output_dir: /workspace/saves/...` stores the output on persistent storage
- Reduced `preprocessing_num_workers` and `dataloader_num_workers` for stability

## Step 4: Running the Training

Create the necessary directories:

```bash
mkdir -p /workspace/cache
mkdir -p /workspace/saves/Qwen2_5-32b-instruct-qa-agent/lora/sft
```

Run the training:

```bash
cd /workspace/Llama-factory-ft-fork/
llamafactory-cli train examples/train_lora/qwen2_5vl_lora_sft.yaml
```

## Troubleshooting

### KeyError: None when processing dataset
- Ensure `dataset_info.json` has the `messages` field set to "conversations"
- Ensure all required fields (`role_tag`, `content_tag`, etc.) are properly set

### Out of disk space error
- Set `cache_dir` in YAML to a location with sufficient space
- Consider using a quantized model to reduce space requirements
- Add the `--offline` flag if you have a local copy of the model

### Memory issues
- Reduce batch size and gradient accumulation steps
- Use a lower LoRA rank (e.g., 4 instead of 8)
- Use 4-bit quantization with `quantization_bit: 4`

## Converting Custom Datasets to ShareGPT Format

If your dataset is in a different format, use this Python script to convert it:

```python
import json

# Input and output files
input_file = "your_original_dataset.json"
output_file = "sharegpt_format.json"

# Load original dataset
with open(input_file, 'r') as f:
    data = json.load(f)

# Convert to ShareGPT format
sharegpt_data = []

# For OpenAI format datasets
for item in data:
    if "messages" in item:
        conversations = []
        for msg in item["messages"]:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            # Map OpenAI roles to ShareGPT roles
            from_role = {
                "system": "system",
                "user": "human",
                "assistant": "gpt"
            }.get(role, role)
            
            # Handle multimodal content
            if isinstance(content, list):
                parts = []
                for part in content:
                    if part.get("type") == "text":
                        parts.append(part.get("text", ""))
                    elif part.get("type") == "image_url":
                        image_tag = f"<img>{part.get('image_url', '')}</img>"
                        parts.append(image_tag)
                content_value = "\n".join(parts)
            else:
                content_value = content
            
            # Add to conversations
            conversations.append({
                "from": from_role,
                "value": content_value
            })
        
        # Add the example if it has conversations
        if conversations:
            sharegpt_data.append({"conversations": conversations})

# Save converted dataset
with open(output_file, 'w') as f:
    json.dump(sharegpt_data, f, indent=2)
```

## Notes on Using Larger GPUs

When using a 160GB GPU:
- You can increase batch size slightly (try 2-4)
- You can increase gradient accumulation steps (16-32)
- You may be able to use 8-bit quantization instead of 4-bit for better quality

Recommended settings for 160GB GPU:
```yaml
per_device_train_batch_size: 2
gradient_accumulation_steps: 16
finetuning_type: lora
lora_rank: 8
```
