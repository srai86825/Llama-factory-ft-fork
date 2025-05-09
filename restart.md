apt-get update && apt-get install -y nano
apt-get update && apt-get install -y tmux
tmux new -s ft
# Install
pip install -r requirements.txt 
pip install bitsandbytes
pip install git+https://github.com/huggingface/transformers
pip install -e .
pip install liger-kernel




# Run 
llamafactory-cli train examples/train_lora/qwen2_5vl_lora_sft.yaml







# FAQ
## 1.  dataset_info files
root@8d0d1ff6f199:/workspace/Llama-factory-ft-fork# find /workspace/Llama-factory-ft-fork/ -type f -name "dataset_info.json"
/workspace/Llama-factory-ft-fork/src/llamafactory/extras/dataset_info.json
/workspace/Llama-factory-ft-fork/src/llamafactory/data/dataset_info.json
/workspace/Llama-factory-ft-fork/dataset_info.json
/workspace/Llama-factory-ft-fork/data/qa_tool_dataset/dataset_info.json
/workspace/Llama-factory-ft-fork/data/dataset_info.json
root@8d0d1ff6f199:/workspace/Llama-factory-ft-fork# 


## 2. Tmux scroll
echo "set -g mouse on" >> ~/.tmux.conf && tmux source-file ~/.tmux.conf


## 3. Saving Ft model
check /saves folder for script and usage

https://huggingface.co/srai86825/qwen-vl-tool-assistant-lora
