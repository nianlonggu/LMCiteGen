import argparse
import os

from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, logging, set_seed
from transformers import Trainer, BitsAndBytesConfig, TrainerCallback

from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset
import numpy as np
import torch

class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if args.should_save:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            kwargs["model"].save_pretrained(checkpoint_path)

            if "pytorch_model.bin" in os.listdir(checkpoint_path):
                os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    
def get_prompt(example, eos_token, index = None):
    """Prepare the text from a sample of the dataset."""
    elements = [
        "\n\n".join(
            [
                "### Citing Paper Title:\n%s"%( example["citing_paper_content"]["title"] ),
                "### Citing Paper Abstract:\n%s"%( example["citing_paper_content"]["abstract"] ),
                "### Cited Paper Title:\n%s"%( example["cited_paper_content"]["title"] ),
                "### Cited Paper Abstract:\n%s"%( example["cited_paper_content"]["abstract"] ),
                "### Text Before Citation:\n%s"%( " ".join( example["text_before_citation"] ) ),
                "### Citation Intent:\n"
            ]
        ),
        example["citation_intent"] + "\n",
        "\n### Keywords:\n",
        "; ".join( example["keywords"] ) + "\n",
        "\n### Citation:\n",
        example["citation"] + " " + " ".join( example["text_after_citation"]  )
    ]
    if index is None:
        index = len(elements)
    text = "".join( elements[:index] )
    return text


def tokenize_example( example ):
    global tokenizer, args
    input_ids = np.array(tokenizer( get_prompt(example, tokenizer.eos_token) ).input_ids)
    labels = np.array([-100] * len(input_ids))
    for index in [1]:
        start_pos = len(tokenizer( get_prompt(example, tokenizer.eos_token, index) ).input_ids)
        end_pos = len(tokenizer( get_prompt(example, tokenizer.eos_token, index+1) ).input_ids)
        ## unmask the labels for intent, keyword and citation sentence
            
    labels[ start_pos: ] = input_ids[start_pos: ]
    
    
    input_ids = input_ids.tolist()[:args.max_length]
    labels  = labels.tolist()[:args.max_length]

    return {"input_ids":input_ids, "labels":labels}


class DataCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id
        
    def __call__(self, examples  ):

        input_ids_list = [ example["input_ids"] for example in examples ]
        labels_list = [ example["labels"] for example in examples ]
        
        padded_length = max( [ len(input_ids) for input_ids in input_ids_list] )
        
        padded_input_ids_list = []
        attention_mask_list = []
        padded_labels_list = []

        for input_ids, labels in zip(input_ids_list, labels_list):
            
            attention_mask = [ 1 ] * len( input_ids ) + [ 0 ] * ( padded_length - len( input_ids ) )
            padded_input_ids = input_ids + [ self.pad_token_id ] * ( padded_length - len(input_ids) )
            padded_labels = labels + [ -100 ] * ( padded_length - len( labels ) )
            
            padded_input_ids_list.append( padded_input_ids )
            attention_mask_list.append( attention_mask )
            padded_labels_list.append( padded_labels )            

        return {
            "input_ids": torch.LongTensor( padded_input_ids_list ),
            "attention_mask": torch.LongTensor( attention_mask_list ),
            "labels": torch.LongTensor( padded_labels_list )
        }



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
    parser.add_argument("--model_path", type=str,  )
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--output_dir", type=str,  )
    parser.add_argument("--streaming", type = int, default = 0 )
    parser.add_argument("--max_steps", type = int, default = 5000, help = "if max_steps is set to -1, then num_train_epochs is effective") 
    parser.add_argument("--num_warmup_steps", type=int, default=500)
    parser.add_argument("--eval_freq", type=int, default=1000 )
    parser.add_argument("--save_freq", type=int, default=1000 )
    parser.add_argument("--log_freq", type=int, default=50 )
    parser.add_argument("--batch_size", type = int, default = 16)
    parser.add_argument("--gradient_accumulation_steps", type = int, default = 1)
    parser.add_argument("--use_lora", type = int, default = 1)
    
    parser.add_argument("--train_dataset_name", type=str, default="data/train.jsonl")
    parser.add_argument("--val_dataset_name", type=str, default="data/val.jsonl")
    parser.add_argument("--quantization", default="int4")
    parser.add_argument("--learning_rate", type = float, default = 1e-4)
    
    parser.add_argument("--shuffle_buffer", type=int, default = 5000)
    parser.add_argument("--max_length", type = int, default = 1024)
    parser.add_argument("--num_train_epochs", type = int, default = 3, help = "To use num_train_epochs, we need to know the total length of the dataset. This is not compatible with the IterableDataset.")
    parser.add_argument("--lr_scheduler_type", type = str, default = "cosine")
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--fp16", type = int, default = 1)
    parser.add_argument("--bf16", type = int, default = 0)
    parser.add_argument("--gradient_checkpointing", type = int, default = 1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lora_r", default=16, type=int)
    parser.add_argument("--lora_alpha", default=32, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    
    args = parser.parse_args()


lora_target_modules_dict = {
    "llama":["q_proj","v_proj"],
    "galactica":["q_proj","v_proj"],
    "gpt-neo":["q_proj","v_proj"],
}
args.lora_target_modules = lora_target_modules_dict[ args.model_type ]


set_seed(args.seed)
os.makedirs(args.output_dir, exist_ok=True)

logging.set_verbosity_error()

tokenizer = get_tokenizer( args.model_path, args.model_type )

train_data = load_dataset('json', 
                        data_files = args.train_dataset_name, 
                        split = 'train',
                        num_proc = args.num_workers if not args.streaming else None,
                        streaming = args.streaming
                 )

if args.streaming:
    train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
else:
    train_data = train_data.shuffle( seed=args.seed)

valid_data = load_dataset('json', 
                        data_files = args.val_dataset_name, 
                        split = 'train',
                        num_proc = args.num_workers if not args.streaming else None,
                        streaming = args.streaming
                 )  

for example in train_data:
    break
exsiting_columns = list(example.keys())

train_dataset = train_data.map( tokenize_example, remove_columns = exsiting_columns )
valid_dataset = valid_data.map( tokenize_example, remove_columns = exsiting_columns )

data_collator = DataCollator(tokenizer.pad_token_id)



print("Loading the model")
  
bnb_config = BitsAndBytesConfig(        
    load_in_8bit= args.quantization == "int8",
    llm_int8_threshold=6.0,
    llm_int8_skip_modules=None,
    llm_int8_enable_fp32_cpu_offload=False,
    llm_int8_has_fp16_weight=False,
    load_in_4bit=args.quantization == "int4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
)

if args.use_lora:
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, quantization_config=bnb_config, device_map={"": Accelerator().process_index}
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules = args.lora_target_modules,
        lora_dropout= args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
else:
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to( "cuda:%d"%( Accelerator().process_index ) )


training_args = TrainingArguments(
    output_dir=args.output_dir,
    dataloader_drop_last=True,
    evaluation_strategy="no" if args.eval_freq <= 0 else "steps",
    max_steps=args.max_steps,
    num_train_epochs=args.num_train_epochs,
    eval_steps=args.eval_freq,
    save_steps=args.save_freq,
    logging_steps=args.log_freq,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    lr_scheduler_type=args.lr_scheduler_type,
    warmup_steps=args.num_warmup_steps,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    gradient_checkpointing=args.gradient_checkpointing,
    fp16=args.fp16,
    bf16=args.bf16,
    weight_decay=args.weight_decay,
    run_name="sft-citation-generator",
    report_to="wandb",
    ddp_find_unused_parameters=False,
)

# optim="paged_adamw_8bit" if args.quantization == "int4" else "adamw_hf", # This paged_adamw_8bit seems to have a negative impact on the training ...

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer = tokenizer,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator = data_collator,
    callbacks= [PeftSavingCallback] if args.use_lora else None
)
        
print_trainable_parameters(trainer.model)

model.config.use_cache = False

print("Training...")
trainer.train()

print("Saving last checkpoint of the model")
trainer.model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))
