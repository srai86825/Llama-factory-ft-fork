import json
import os

# Input and output files
input_file = "/workspace/LLaMA-Factory/data/qa_tool_dataset/dataset.json"
output_file = "/workspace/LLaMA-Factory/data/qa_tool_dataset/alpaca_data_cleaned.json"

def convert_to_alpaca():
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
        
        # Handle different potential structures
        if isinstance(data, list):
            for idx, item in enumerate(data):
                print(f"Processing item {idx}...")
                
                # Case 1: Direct "conversations" array
                if "conversations" in item:
                    conversations = item["conversations"]
                    print(f"  Found conversations array with {len(conversations)} messages")
                    
                    # Extract messages
                    human_msgs = [msg["value"] for msg in conversations if msg.get("from") == "human"]
                    gpt_msgs = [msg["value"] for msg in conversations if msg.get("from") == "gpt"]
                    
                    print(f"  Found {len(human_msgs)} human messages and {len(gpt_msgs)} gpt messages")
                    
                    if human_msgs and gpt_msgs:
                        alpaca_item = {
                            "instruction": human_msgs[0],
                            "input": "",
                            "output": gpt_msgs[0]
                        }
                        alpaca_data.append(alpaca_item)
                        print(f"  ✓ Added alpaca entry")
                
                # Case 2: "messages" array (OpenAI format)
                elif "messages" in item:
                    messages = item["messages"]
                    print(f"  Found messages array with {len(messages)} messages")
                    
                    # Extract messages by role
                    user_msgs = []
                    assistant_msgs = []
                    
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
                                user_msgs.append("\n".join(parts))
                            else:
                                user_msgs.append(content)
                        
                        elif role == "assistant":
                            assistant_msgs.append(content)
                    
                    print(f"  Found {len(user_msgs)} user messages and {len(assistant_msgs)} assistant messages")
                    
                    if user_msgs and assistant_msgs:
                        alpaca_item = {
                            "instruction": user_msgs[0],
                            "input": "",
                            "output": assistant_msgs[0]
                        }
                        alpaca_data.append(alpaca_item)
                        print(f"  ✓ Added alpaca entry")
        
        print(f"Conversion complete. Generated {len(alpaca_data)} alpaca items.")
        
        # Write alpaca format data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(alpaca_data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved to {output_file}")
        
        # If no items were converted, create a sample
        if len(alpaca_data) == 0:
            print("WARNING: No items were converted. Creating a sample alpaca item for debugging.")
            sample_file = "/workspace/LLaMA-Factory/data/qa_tool_dataset/sample_alpaca.json"
            sample_data = [
                {
                    "instruction": "What is in this image?<img>https://example.com/sample.jpg</img>",
                    "input": "",
                    "output": "The image shows a sample landscape with mountains and a lake."
                }
            ]
            with open(sample_file, 'w', encoding='utf-8') as f:
                json.dump(sample_data, f, ensure_ascii=False, indent=2)
            print(f"Sample saved to {sample_file}")
            
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    convert_to_alpaca()