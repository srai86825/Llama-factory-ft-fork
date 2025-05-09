import json
import os

# Input and output files
input_file = "dataset.json"
output_file = "dataset_fixed_sharegpt_v1.json"

def fix_dataset():
    try:
        # Read original dataset
        print(f"Reading dataset from: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Dataset loaded. Type: {type(data)}, Length: {len(data) if isinstance(data, list) else 'N/A'}")
        if isinstance(data, list) and len(data) > 0:
            print(f"First item structure keys: {list(data[0].keys())}")
            if "conversations" in data[0]:
                print(f"First conversation: {data[0]['conversations'][0]}")
        
        # Ensure valid ShareGPT format
        fixed_data = []
        for idx, item in enumerate(data):
            print(f"Processing item {idx}...")
            
            # Check if item has conversations field
            if "conversations" in item:
                conversations = item["conversations"]
                # Check if conversations is a list
                if isinstance(conversations, list):
                    # Check if each conversation has from and value
                    valid = True
                    for conv in conversations:
                        if not isinstance(conv, dict) or "from" not in conv or "value" not in conv:
                            valid = False
                            print(f"  Invalid conversation format: {conv}")
                            break
                    
                    if valid:
                        fixed_data.append(item)
                        print(f"  ✓ Item is valid")
                    else:
                        print(f"  ✗ Item has invalid conversations")
                else:
                    print(f"  ✗ 'conversations' is not a list")
            # If item has messages, convert to conversations
            elif "messages" in item:
                messages = item["messages"]
                conversations = []
                
                for msg in messages:
                    role = msg.get("role")
                    content = msg.get("content")
                    
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
                    
                    conversations.append({
                        "from": from_role,
                        "value": content_value
                    })
                
                if conversations:
                    fixed_data.append({"conversations": conversations})
                    print(f"  ✓ Converted messages to conversations")
            else:
                print(f"  ✗ Item has neither conversations nor messages")
        
        # If no items were fixed, create a sample dataset
        if len(fixed_data) == 0:
            print("No valid items found. Creating a sample dataset.")
            fixed_data = [
                {
                    "conversations": [
                        {"from": "system", "value": "You are a helpful visual assistant."},
                        {"from": "human", "value": "What is shown in this image?<img>https://example.com/sample.jpg</img>"},
                        {"from": "gpt", "value": "The image shows a mountain landscape with a lake and trees in the foreground."}
                    ]
                },
                {
                    "conversations": [
                        {"from": "system", "value": "You are a helpful visual assistant."},
                        {"from": "human", "value": "Describe what you see in this chart.<img>https://example.com/chart.jpg</img>"},
                        {"from": "gpt", "value": "This is a bar chart showing sales data for different product categories. The 'Electronics' category has the highest sales."}
                    ]
                }
            ]
            print(f"Added {len(fixed_data)} sample examples")
        
        # Write the result to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(fixed_data, f, ensure_ascii=False, indent=2)
        
        print(f"Fixed {len(fixed_data)} examples and saved to {output_file}")
        
        # Make a backup of the original file
        backup_file = f"{input_file}.bak"
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Backup of original dataset saved to {backup_file}")
        
        # Replace the original file with the fixed one
        os.rename(output_file, input_file)
        print(f"Original dataset replaced with fixed version")
        
        # Print a sample to verify
        if len(fixed_data) > 0:
            print("\nSample entry:")
            print(json.dumps(fixed_data[0], indent=2))
        
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    fix_dataset()