import json
import os

# Input and output file paths
input_file = "original_dataset.json"  # Your original dataset
output_file = "dataset.json"          # The converted dataset

def convert_dataset():
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # New format required by LLaMA-Factory
    converted_data = []
    
    for item in data:
        messages = item["messages"]
        conversations = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "user":
                # Handle multimodal content
                if isinstance(content, list):
                    parts = []
                    for part in content:
                        if part["type"] == "text":
                            parts.append(part["text"])
                        elif part["type"] == "image_url":
                            # For Qwen2.5-VL, we'll use a special tag format
                            # This will be processed during training to load the actual image
                            image_tag = f"<img>{part['image_url']}</img>"
                            parts.append(image_tag)
                    combined = "\n".join(parts)
                    conversations.append({"from": "human", "value": combined})
                else:
                    # Handle text-only content
                    conversations.append({"from": "human", "value": content})
            
            elif role == "assistant":
                # Assistant content is typically text-only
                conversations.append({"from": "gpt", "value": content})
            
            elif role == "system":
                # System messages can be prepended to the first human message
                # or handled as a special conversation turn
                conversations.append({"from": "gpt", "value": content})
        
        # Only add examples with valid conversation structure
        if len(conversations) >= 2:  # At least one human and one assistant message
            converted_data.append({"conversations": conversations})
    
    # Write the converted data
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print(f"Converted {len(converted_data)} examples to {output_file}")

if __name__ == "__main__":
    convert_dataset()