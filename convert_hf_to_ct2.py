import os
from transformers import AutoTokenizer
import argparse
import ctranslate2
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,  )
    parser.add_argument("--output_dir", type=str,  )
    parser.add_argument("--quantization", default = None, help="can be int8, float16 and int8_float16, if it is not set, then do not apply quantization"  )
    parser.add_argument("--load_as_float16", action="store_true" )
    parser.add_argument("--low_cpu_mem_usage", action="store_true" )
    parser.add_argument("--trust_remote_code", action="store_true" )
    parser.add_argument("--force", action="store_true" )
    parser.add_argument("--model_type")
    
    args = parser.parse_args()
    
    supported_model_types = ["galactica", "llama"]
    assert args.model_type in supported_model_types, "Supported model types: %s"%( str(supported_model_types) )
    
    converter = ctranslate2.converters.TransformersConverter(
        model_name_or_path = args.model,
        load_as_float16 = args.load_as_float16,
        low_cpu_mem_usage = args.low_cpu_mem_usage,
        trust_remote_code = args.trust_remote_code
    )
    
    converter.convert(
        output_dir = args.output_dir, 
        quantization = args.quantization, 
        force = args.force
    )
    
    ## save original tokenizer
    tokenizer = AutoTokenizer.from_pretrained( args.model )
    tokenizer.save_pretrained( args.output_dir )