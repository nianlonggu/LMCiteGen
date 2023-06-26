from accelerate import Accelerator
from reward_model.citation_intent_classification.src.BertClassifier.model import CitationIntentClassifier
import json
from tqdm import tqdm
import evaluate
from rouge_score import rouge_scorer
from transformers import Trainer, AutoConfig, AutoModelForCausalLM, AutoTokenizer,  \
                          TrainingArguments, logging, \
                          BitsAndBytesConfig, TrainerCallback
import torch
import numpy as np


from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import argparse
import os
from nltk.tokenize import sent_tokenize

import time


class RewardCal:
    def __init__(self, intent_classifier_model_path, pretrained_lm_path, device_index ):
        self.rouge = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
        self.intent_classifier = CitationIntentClassifier( intent_classifier_model_path,  "allenai/scibert_scivocab_uncased",  device_index )
        self.lm = AutoModelForCausalLM.from_pretrained(pretrained_lm_path, load_in_4bit = True, device_map={"":device_index})
        self.lm.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_lm_path)
        
        with torch.no_grad():
            log_prior_distribution = self.lm(input_ids=torch.LongTensor( [ self.tokenizer.bos_token_id ] ).unsqueeze(0)
                   )["logits"].to(torch.float32).log_softmax(-1)[0,0].detach().cpu().numpy()
        self.log_prior_distribution = log_prior_distribution
    
    def sigmoid(self, x):
        return 1/(1+np.exp( -x ))
        
    def reward_fn(self, gen_citation,  given_intent, given_keywords, given_citation  ):
        if gen_citation.strip() == "":
            return 0.0
        
        reward_intent = self.intent_classifier.get_intent_scores( gen_citation ).get(given_intent, 0.0)
        reward_keywords = self.rouge.score( given_keywords, gen_citation )["rougeL"].recall
        
        cit_token_ids = np.array(self.tokenizer.encode( gen_citation ))
        prior_cit_log_probs = self.log_prior_distribution[ cit_token_ids ]
        
        input_ids = torch.LongTensor([self.tokenizer.bos_token_id] + cit_token_ids.tolist() ).unsqueeze(0)
        with torch.no_grad():
            cit_log_probs = self.lm(input_ids = input_ids)["logits"][0][:-1].to(torch.float32).log_softmax(-1).detach().cpu().numpy()
            cit_log_probs = cit_log_probs[ np.arange(len(cit_token_ids)), cit_token_ids ]
        reward_fluency = self.sigmoid( (np.mean(cit_log_probs - prior_cit_log_probs ) - 4) )
        # reward_fluency = np.mean(cit_log_probs - prior_cit_log_probs )
        
        reward_groundness = self.rouge.score( given_citation, gen_citation )
        reward_groundness = reward_groundness["rouge1"].fmeasure + reward_groundness["rouge2"].fmeasure + reward_groundness["rougeL"].fmeasure 
        
        
        return (reward_intent + reward_keywords + reward_fluency + reward_groundness)/4
        


def build_dataset( tokenizer, dataset_name, input_min_text_length, input_max_text_length, num_workers ):
    train_data = load_dataset('json', 
                            data_files = dataset_name, 
                            split = 'train',
                            num_proc = num_workers  )

    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for pos in range( len(examples["citing_id"] ) ): 
            query = "\n\n".join(
                [
                    "### Citing Paper Title:\n%s"%( examples["citing_paper_content"][pos]["title"] ),
                    "### Citing Paper Abstract:\n%s"%( examples["citing_paper_content"][pos]["abstract"] ),
                    "### Cited Paper Title:\n%s"%( examples["cited_paper_content"][pos]["title"] ),
                    "### Cited Paper Abstract:\n%s"%( examples["cited_paper_content"][pos]["abstract"] ),
                    "### Text Before Citation:\n%s"%( " ".join( examples["text_before_citation"][pos] ) ),
                    "### Citation Intent:\n%s"%( examples["citation_intent"][pos] ),
                    "### Keywords:\n%s"%( "; ".join( examples["keywords"][pos] ) ),
                    "### Citation:\n"
                ]
            )
            query_input_ids = tokenizer( query, truncation = True ).input_ids
            new_examples["input_ids"].append( query_input_ids )
            new_examples["query"].append( 
                                            {
                                                "citation_intent":examples["citation_intent"][pos],
                                                "keywords":"; ".join( examples["keywords"][pos] ),
                                                "citation":examples["citation"][pos] 
                                            }
                                         
                                        )
        return new_examples
    
    for example in train_data:
        break
    original_columns = list( example.keys() )

    ds = train_data.map(
        preprocess_function,
        batched=True,
        num_proc=args.num_workers,
        remove_columns=original_columns,
    )
    # filter (to keep) examples that match the length criteria
    ds = ds.filter(lambda x: len(x["input_ids"]) >= input_min_text_length and len(x["input_ids"]) <= input_max_text_length, batched=False)
    ds.set_format(type="torch")
    return ds



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

def get_citation_text( response_text, min_sen_length = 15 ):
    
    sens = response_text.split(". ")
    cit_sen = ""
    for count, sen in enumerate( sens ):
        cit_sen += sen + (". " if count < len(sens)-1 else "" )
        if len(cit_sen.split()) >= min_sen_length:
            break
    if cit_sen.endswith(". "):
        cit_sen = cit_sen[:-1]
    return cit_sen

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])    



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
    parser.add_argument("--model_path", type=str )
    parser.add_argument("--model_type", type=str )
    parser.add_argument("--output_dir", type=str )
    parser.add_argument("--train_dataset_name", type=str, default="data/train.jsonl")

    parser.add_argument("--save_freq", type=int, default=100 )    
    parser.add_argument("--batch_size", type = int, default = 32)
    parser.add_argument("--mini_batch_size", type = int, default = 1)
    parser.add_argument("--gradient_accumulation_steps", type = int, default = 8)

    parser.add_argument("--quantization", default="int4", type=str)

    parser.add_argument("--intent_classifier_model_path", type=str, default="reward_model/citation_intent_classification/model/BertClassifier/5_5_0.05_0.01/model_batch_515.pt")
    parser.add_argument("--reward_lm_model_path", type=str, default="bigscience/bloom-560m")
    parser.add_argument("--input_min_text_length", type=int, default=500 )
    parser.add_argument("--input_max_text_length", type=int, default=900 )
    parser.add_argument("--num_workers", type=int, default=8 )
    parser.add_argument("--seed", type=int, default=0 )
    parser.add_argument("--learning_rate", type=float, default=1.41e-5 )
    parser.add_argument("--log_with", type=str, default="wandb" )

    parser.add_argument("--max_steps", type=int, default = 100000)
    parser.add_argument("--target_kl", type=float, default=0.1 )
    parser.add_argument("--ppo_epochs", type = int, default = 4)
    parser.add_argument("--reward_baseline", type=float, default = 0. )
    parser.add_argument("--lora_r", default=16, type=int)
    parser.add_argument("--lora_alpha", default=32, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
        
    args = parser.parse_args()


# In[7]:

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

train_dataset = build_dataset( tokenizer, args.train_dataset_name, 
                           args.input_min_text_length, args.input_max_text_length, 
                           args.num_workers )


# In[8]:





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

current_device = Accelerator().local_process_index
pretrained_model = AutoModelForCausalLM.from_pretrained(
    args.model_path, quantization_config=bnb_config, device_map={"": current_device}
)
pretrained_model = prepare_model_for_kbit_training(pretrained_model)


lora_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    target_modules = args.lora_target_modules,
    lora_dropout= args.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)

pretrained_model = get_peft_model(pretrained_model, lora_config)

model = AutoModelForCausalLMWithValueHead.from_pretrained( pretrained_model )


print_trainable_parameters( model )


ppo_config = PPOConfig(
    steps=args.max_steps,
    model_name=args.model_path,
    learning_rate=args.learning_rate,
    log_with=args.log_with,
    batch_size=args.batch_size,
    mini_batch_size=args.mini_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    optimize_cuda_cache=True,
    early_stopping=False,
    target_kl=args.target_kl,
    ppo_epochs=args.ppo_epochs,
    seed=args.seed,
    init_kl_coef=0.2,
    adap_kl_ctrl=True
)
ppo_trainer = PPOTrainer(ppo_config, 
                         model, 
                         ref_model=None, 
                         tokenizer=tokenizer, 
                         dataset=train_dataset, 
                         data_collator=collator,
                         optimizer = None
                        )


# In[12]:


reward_cal = RewardCal( args.intent_classifier_model_path, args.reward_lm_model_path, current_device )


# In[13]:


generation_kwargs = {
    # "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": 100_000,
}
output_min_length = 40
output_max_length = 60
output_length_sampler = LengthSampler(output_min_length, output_max_length)


# In[14]:


for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    if epoch >= ppo_config.total_ppo_epochs:
        break
        
    question_tensors = batch["input_ids"]

    print("Sampling ... ")
    tic = time.time()

    response_tensors = ppo_trainer.generate(
        question_tensors,
        return_prompt=False,
        length_sampler=output_length_sampler,
        **generation_kwargs,
    )
    print("Sampling time: %d s"%(time.time() - tic ))

    batch["response"] = [ get_citation_text(sen) for sen in tokenizer.batch_decode(response_tensors, skip_special_tokens=True)]
    rewards = []
    for idx in range( len(batch["response"]) ):
        r = reward_cal.reward_fn( batch["response"][idx],  
                                  batch["query"][idx]["citation_intent"],
                                  batch["query"][idx]["keywords"],
                                  batch["query"][idx]["citation"]
                                )
        rewards.append(  torch.tensor(  r - args.reward_baseline , dtype = torch.float32  ) )
    
    print("PPOing ... ")
    tic = time.time()

    # Run PPO step
    stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

    print("PPO time: %d s \n"%(time.time() - tic ))

    if args.save_freq and epoch and epoch % args.save_freq == 0:
        ppo_trainer.save_pretrained(args.output_dir + f"/checkpoint-{epoch}")
    



