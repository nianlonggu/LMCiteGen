from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import BartForConditionalGeneration
import torch
for model_path, remote_path in [
    ( "../sft_model/galactica-125m-hf/checkpoint-5000", "nianlong/citgen-galactica-125m-sft" ),
    ("../ppo_model/galactica-125m-hf", "nianlong/citgen-galactica-125m-ppo"  ),
    ("../sft_model/galactica-1.3b-hf/checkpoint-3000", "nianlong/citgen-galactica-1.3b-sft" ),
    ("../sft_model/galactica-6.7b-hf", "nianlong/citgen-galactica-6.7b-sft" ),
    ("../ppo_model/galactica-6.7b-hf", "nianlong/citgen-galactica-6.7b-ppo" ),
    ("../sft_model/gpt-neo-125m-hf/checkpoint-5000", "nianlong/citgen-gpt-neo-125m-sft"  ),
    ("../sft_model/gpt-neo-1.3b-hf/checkpoint-5000", "nianlong/citgen-gpt-neo-1.3b-sft" ),
    ("../sft_model/llama-7b-hf", "nianlong/citgen-llama-7b-sft" ),
    ("../ppo_model/llama-7b-hf", "nianlong/citgen-llama-7b-ppo" ),
    ("../sft_model/bart-base/checkpoint-5000", "nianlong/citgen-bart-base"),
    ("../sft_model/bart-large/checkpoint-5000", "nianlong/citgen-bart-large")
]:
    print(model_path, remote_path)
    tokenizer = AutoTokenizer.from_pretrained( model_path )
    
    if "bart-" in model_path:
        model = BartForConditionalGeneration.from_pretrained( model_path, torch_dtype=torch.float16 )
    else:
        model = AutoModelForCausalLM.from_pretrained( model_path, torch_dtype=torch.float16 )
    
    tokenizer.push_to_hub( remote_path )
    model.push_to_hub( remote_path )