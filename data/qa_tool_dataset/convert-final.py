import json
import os

# Input and output files
input_file = "dataset.json"
output_file = "alpaca_data_cleaned.json"

def convert_to_sharegpt():
    try:
        # Read original dataset
        print(f"Reading dataset from: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Dataset loaded. Type: {type(data)}, Length: {len(data) if isinstance(data, list) else 'N/A'}")
        
        # Convert to ShareGPT format
        sharegpt_data = []
        
        for idx, item in enumerate(data):
            print(f"Processing item {idx}...")
            
            # Handle dataset with "messages" array (OpenAI format)
            if "messages" in item:
                conversations = []
                messages = item["messages"]
                print(f"  Found messages array with {len(messages)} messages")
                
                for msg in messages:
                    role = msg.get("role")
                    content = msg.get("content")
                    
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
                
                # Only add if we have conversations
                if conversations:
                    sharegpt_item = {
                        "conversations": conversations
                    }
                    sharegpt_data.append(sharegpt_item)
                    print(f"  ✓ Added ShareGPT entry with {len(conversations)} messages")
            
            # Handle dataset with "conversations" array (already in ShareGPT format)
            elif "conversations" in item:
                sharegpt_data.append(item)
                print(f"  ✓ Item already in ShareGPT format with {len(item['conversations'])} messages")
        
        # If no items were converted, create a sample dataset
        if len(sharegpt_data) == 0:
            print("No items were properly converted. Adding fallback examples.")
            sharegpt_data = [
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
            print(f"Added {len(sharegpt_data)} fallback examples")
        
        # Write the result to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sharegpt_data, f, ensure_ascii=False, indent=2)
        
        print(f"Converted {len(sharegpt_data)} examples and saved to {output_file}")
        
        # Print a sample to verify
        if len(sharegpt_data) > 0:
            print("\nSample entry:")
            print(json.dumps(sharegpt_data[0], indent=2))
        
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    convert_to_sharegpt()