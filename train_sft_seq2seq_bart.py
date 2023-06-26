#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


from transformers import BartForConditionalGeneration


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
    
def get_prompt(example):
    """Prepare the text from a sample of the dataset."""
    
    input_text = "\n\n".join(
            [
                "### Citing Paper Title:\n%s"%( example["citing_paper_content"]["title"] ),
                "### Citing Paper Abstract:\n%s"%( example["citing_paper_content"]["abstract"] ),
                "### Cited Paper Title:\n%s"%( example["cited_paper_content"]["title"] ),
                "### Cited Paper Abstract:\n%s"%( example["cited_paper_content"]["abstract"] ),
                "### Text Before Citation:\n%s"%( " ".join( example["text_before_citation"] ) )
            ]
        )
    output_text = "### Citation Intent:\n%s\n\n### Keywords:\n%s\n\n### Citation:\n%s" % (
            example["citation_intent"],
            "; ".join( example["keywords"] ),
            example["citation"] + " " + " ".join( example["text_after_citation"]  )
    )
    return input_text, output_text

def tokenize_example( example ):
    global tokenizer, args

    input_text, output_text = get_prompt(example)
    input_ids = np.array( tokenizer.encode( input_text, max_length=args.max_length, truncation=True ) )
    
    decoder_input_ids = np.array( tokenizer.encode( output_text, max_length=args.max_gen_length, truncation= True) )
    labels = decoder_input_ids[1:]
    decoder_input_ids = decoder_input_ids[:-1]
    
    return {
        "input_ids":input_ids.tolist() ,
        "decoder_input_ids":decoder_input_ids.tolist() ,
        "labels":labels.tolist()
    }


class DataCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id
        
    def __call__(self, examples  ):

        input_ids_list = [ example["input_ids"] for example in examples ]
        decoder_input_ids_list = [ example["decoder_input_ids"]  for example in examples ]
        labels_list = [ example["labels"] for example in examples ]
        
        padded_input_ids_length = max( [ len(input_ids) for input_ids in input_ids_list] )
        padded_decoder_length = max( [ len(input_ids) for input_ids in decoder_input_ids_list] )
        
        padded_input_ids_list = []
        attention_mask_list = []
        padded_decoder_input_ids_list = []
        decoder_attention_mask_list = []
        padded_labels_list = []

        for input_ids, decoder_input_ids, labels in zip(input_ids_list, decoder_input_ids_list, labels_list):
            attention_mask = [ 1 ] * len( input_ids ) + [ 0 ] * ( padded_input_ids_length - len( input_ids ) )
            padded_input_ids = input_ids + [ self.pad_token_id ] * ( padded_input_ids_length - len(input_ids) )
            
            decoder_attention_mask = [ 1 ] * len( decoder_input_ids ) + [ 0 ] * ( padded_decoder_length - len( decoder_input_ids ) )
            padded_decoder_input_ids = decoder_input_ids + [ self.pad_token_id ] * ( padded_decoder_length - len( decoder_input_ids ) )
            
            padded_labels = labels + [ -100 ] * ( padded_decoder_length - len( labels ) )
                
            padded_input_ids_list.append( padded_input_ids )
            attention_mask_list.append( attention_mask )
            padded_decoder_input_ids_list.append( padded_decoder_input_ids )
            decoder_attention_mask_list.append( decoder_attention_mask )
            padded_labels_list.append( padded_labels )
         

        return {
            "input_ids": torch.LongTensor( padded_input_ids_list ),
            "attention_mask": torch.LongTensor( attention_mask_list ),
            "decoder_input_ids": torch.LongTensor( padded_decoder_input_ids_list ),
            "decoder_attention_mask": torch.LongTensor( decoder_attention_mask_list ),

            "labels": torch.LongTensor( padded_labels_list )
        }



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,  )
    parser.add_argument("--output_dir", type=str,  )
    parser.add_argument("--streaming", type = int, default = 0 )
    parser.add_argument("--log_freq", type=int, default=50 )
    parser.add_argument("--batch_size", type = int, default = 16)
    parser.add_argument("--gradient_accumulation_steps", type = int, default = 1)
    parser.add_argument("--max_steps", type = int, default = 5000, help = "if max_steps is set to -1, then num_train_epochs is effective") 
    parser.add_argument("--eval_freq", type=int, default=1000 )
    parser.add_argument("--save_freq", type=int, default=1000 )
    parser.add_argument("--train_dataset_name", type=str, default="data/train.jsonl")
    parser.add_argument("--val_dataset_name", type=str, default="data/val.jsonl")    
    parser.add_argument("--shuffle_buffer", type=int, default = 5000)
    parser.add_argument("--max_length", type = int, default = 1024)
    parser.add_argument("--max_gen_length", type = int, default = 200)
    parser.add_argument("--num_train_epochs", type = int, default = 3, help = "To use num_train_epochs, we need to know the total length of the dataset. This is not compatible with the IterableDataset.")
    parser.add_argument("--learning_rate", type = float, default = 1e-5)
    parser.add_argument("--lr_scheduler_type", type = str, default = "cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--fp16", type = int, default = 1)
    parser.add_argument("--bf16", type = int, default = 0)
    parser.add_argument("--gradient_checkpointing", type = int, default = 1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=8)
    
    args = parser.parse_args()


# In[3]:


# import re

# para_matcher  = re.compile('"--(.*?)"')
# value_matcher = re.compile('default\s*=\s*(.*?)[\s\),]')

# for line in """
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_path", type=str,  )
#     parser.add_argument("--output_dir", type=str,  )
#     parser.add_argument("--streaming", type = int, default = 0 )
#     parser.add_argument("--log_freq", type=int, default=50 )
#     parser.add_argument("--batch_size", type = int, default = 16 )
#     parser.add_argument("--gradient_accumulation_steps", type = int, default = 1 )
#     parser.add_argument("--max_steps", type = int, default = 5000, help = "if max_steps is set to -1, then num_train_epochs is effective") 
#     parser.add_argument("--eval_freq", type=int, default=1000 )
#     parser.add_argument("--save_freq", type=int, default=1000 )
#     parser.add_argument("--train_dataset_name", type=str, default="data/train_sft.jsonl" )
#     parser.add_argument("--val_dataset_name", type=str, default="data/val.jsonl" )    
#     parser.add_argument("--shuffle_buffer", type=int, default = 5000 )
#     parser.add_argument("--max_length", type = int, default = 1024 )
#     parser.add_argument("--num_train_epochs", type = int, default = 3, help = "To use num_train_epochs, we need to know the total length of the dataset. This is not compatible with the IterableDataset.")
#     parser.add_argument("--learning_rate", type = float, default = 1e-4 )
#     parser.add_argument("--lr_scheduler_type", type = str, default = "cosine" )
#     parser.add_argument("--num_warmup_steps", type=int, default=500 )
#     parser.add_argument("--weight_decay", type=float, default=0.05 )
#     parser.add_argument("--local_rank", type=int, default=0 )
#     parser.add_argument("--fp16", type = int, default = 1 )
#     parser.add_argument("--bf16", type = int, default = 0 )
#     parser.add_argument("--gradient_checkpointing", type = int, default = 1 )
#     parser.add_argument("--seed", type=int, default=0 )
#     parser.add_argument("--num_workers", type=int, default=8 )
#     """.split("\n"):
#     para = (para_matcher.findall(line)+[None])[0]
#     if para is not None:
#         value = (value_matcher.findall(line) + [""])[0]
#         print("args.%s = %s"%( para, value ))


# In[4]:


set_seed(args.seed)
os.makedirs(args.output_dir, exist_ok=True)

logging.set_verbosity_error()

tokenizer = AutoTokenizer.from_pretrained(args.model_path)

train_data = load_dataset('json', 
                        data_files = args.train_dataset_name, 
                        split = 'train',
                        num_proc = args.num_workers if not args.streaming else None,
                        streaming = args.streaming
                 )


# In[5]:


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


# In[6]:


for example in train_data:
    break
exsiting_columns = list(example.keys())


# In[7]:


train_dataset = train_data.map( tokenize_example, remove_columns = exsiting_columns )
valid_dataset = valid_data.map( tokenize_example, remove_columns = exsiting_columns )


# In[8]:


data_collator = DataCollator(tokenizer.pad_token_id)


# In[9]:


print("Loading the model")


# In[10]:


model = BartForConditionalGeneration.from_pretrained(args.model_path).to( "cuda:%d"%( Accelerator().process_index ) )


# In[11]:


print_trainable_parameters( model)


# In[12]:


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
    run_name="sft-citation-generator-seq2seq",
    report_to="wandb",
    ddp_find_unused_parameters=False,
)


# In[13]:


trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer = tokenizer,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator = data_collator,
)


# In[ ]:


print("Training...")
trainer.train()


# In[ ]:


print("Saving last checkpoint of the model")
trainer.model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))


# In[ ]:




