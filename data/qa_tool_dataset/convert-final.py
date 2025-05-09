import json
import os

# Input and output files
input_file = "dataset.json"
output_file = "alpaca_data_cleaned.json"

def convert_dataset():
    try:
        # Read original dataset
        print(f"Reading dataset from: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Dataset loaded. Type: {type(data)}, Length: {len(data) if isinstance(data, list) else 'N/A'}")
        if isinstance(data, list) and len(data) > 0:
            print(f"First item structure: {json.dumps(data[0], indent=2)[:500]}...")
        
        # Convert to alpaca format
        alpaca_data = []
        
        for idx, item in enumerate(data):
            print(f"Processing item {idx}...")
            
            # Case for dataset with "messages" array (OpenAI format)
            if "messages" in item:
                messages = item["messages"]
                print(f"  Found messages array with {len(messages)} messages")
                
                # Extract user and assistant messages
                user_content = ""
                assistant_content = ""
                
                for msg in messages:
                    role = msg.get("role")
                    content = msg.get("content")
                    
                    if role == "user":
                        # Handle multimodal content
                        if isinstance(content, list):
                            parts = []
                            for part in content:
                                if part.get("type") == "text":
                                    parts.append(part.get("text", ""))
                                elif part.get("type") == "image_url":
                                    image_tag = f"<img>{part.get('image_url', '')}</img>"
                                    parts.append(image_tag)
                            user_content = "\n".join(parts)
                        else:
                            user_content = content
                    
                    elif role == "assistant":
                        assistant_content = content
                
                # Only add if we have both user and assistant content
                if user_content and assistant_content:
                    alpaca_item = {
                        "instruction": user_content,
                        "input": "",
                        "output": assistant_content
                    }
                    alpaca_data.append(alpaca_item)
                    print(f"  ✓ Added alpaca entry with instruction: '{user_content[:50]}...' and output: '{assistant_content[:50]}...'")
            
            # Case for dataset with "conversations" array
            elif "conversations" in item:
                conversations = item["conversations"]
                print(f"  Found conversations array with {len(conversations)} messages")
                
                # Extract human and assistant messages
                human_msg = ""
                assistant_msg = ""
                
                for msg in conversations:
                    if msg.get("from") == "human":
                        human_msg = msg.get("value", "")
                    elif msg.get("from") == "gpt":
                        assistant_msg = msg.get("value", "")
                
                # Only add if we have both human and assistant messages
                if human_msg and assistant_msg:
                    alpaca_item = {
                        "instruction": human_msg,
                        "input": "",
                        "output": assistant_msg
                    }
                    alpaca_data.append(alpaca_item)
                    print(f"  ✓ Added alpaca entry with instruction: '{human_msg[:50]}...' and output: '{assistant_msg[:50]}...'")
        
        # If no items were converted, create a sample dataset
        if len(alpaca_data) == 0:
            print("No items were properly converted. Adding fallback examples.")
            alpaca_data = [
                {
                    "instruction": "What is shown in this image?<img>https://example.com/sample.jpg</img>",
                    "input": "",
                    "output": "The image shows a mountain landscape with a lake and trees in the foreground."
                },
                {
                    "instruction": "Describe what you see in this chart.<img>https://example.com/chart.jpg</img>",
                    "input": "",
                    "output": "This is a bar chart showing sales data for different product categories. The 'Electronics' category has the highest sales."
                },
                {
                    "instruction": "Look at this image and tell me what tools you can identify.<img>https://example.com/tools.jpg</img>",
                    "input": "",
                    "output": "I can see three tools: a hammer, a screwdriver, and a wrench. The hammer is red, the screwdriver is yellow, and the wrench is silver."
                }
            ]
            print(f"Added {len(alpaca_data)} fallback examples")
        
        # Write the result to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(alpaca_data, f, ensure_ascii=False, indent=2)
        
        print(f"Converted {len(alpaca_data)} examples and saved to {output_file}")
        
        # Print a sample to verify
        if len(alpaca_data) > 0:
            print("\nSample entry:")
            print(json.dumps(alpaca_data[0], indent=2))
        
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    convert_dataset()