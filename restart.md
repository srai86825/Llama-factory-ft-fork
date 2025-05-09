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