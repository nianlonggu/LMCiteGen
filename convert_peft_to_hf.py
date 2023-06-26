import argparse
import os

import torch
import transformers
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, logging, set_seed


def get_tokenizer( model_path, model_type ):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if model_type == "llama" or model_type == "gpt-neo":
        if tokenizer.eos_token is None:
            print("Error: no eos token pre-defined in the vocabulary. You need to add the eos_token and resize the model's embedding accordingly")
            assert False
        
        special_tokens ={}
        if tokenizer.bos_token is None:
            special_tokens["bos_token"] = tokenizer.eos_token
        if tokenizer.unk_token is None:
            special_tokens["unk_token"] = tokenizer.eos_token
        if tokenizer.pad_token is None:
            special_tokens["pad_token"] = tokenizer.unk_token
    
        tokenizer.add_special_tokens( special_tokens )
    elif model_type == "galactica":
        if tokenizer.eos_token is None:
            tokenizer.add_special_tokens( {
                "bos_token":"<s>",
                "eos_token":"</s>",
                "pad_token":"<pad>",
                "unk_token":"<unk>"
            } )
            
    else:
        print("Unsupported model type!")
        assert False
        
    return tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path" )
    parser.add_argument("--lora_model_path" )
    parser.add_argument("--save_model_path" )
    parser.add_argument("--model_type")
    
    args = parser.parse_args()
    
    supported_model_types = ["galactica", "llama"]
    assert args.model_type in supported_model_types, "Supported model types: %s"%( str(supported_model_types) )
    
    tokenizer = get_tokenizer( args.base_model_path , args.model_type )
        
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
    )
    
    lora_model = PeftModel.from_pretrained(
        base_model,
        args.lora_model_path,
        device_map={"": "cpu"},
        torch_dtype=torch.float16,
    )
    
    # merge weights - new merging method from peft
    lora_model = lora_model.merge_and_unload()
    lora_model.train(False)
    
    lora_model_sd = lora_model.state_dict()
    deloreanized_sd = {
        k.replace("base_model.model.", ""): v
        for k, v in lora_model_sd.items()
        if "lora" not in k
    }

    base_model.save_pretrained( args.save_model_path, state_dict=deloreanized_sd )
    tokenizer.save_pretrained( args.save_model_path )
    
    