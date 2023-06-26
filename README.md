## Hardware Requirement
This script has been tested on a GPU server with two RTX A6000 GPUs. 
* OS: Ubuntu 22.04
* CUDA Version: 11.7

## Setup Environment and Install Dependency
* Insatll anaconda https://docs.conda.io/en/latest/miniconda.html  
* Create an anaconda environment:
```bash
conda create -n citgen python=3.10
```
<br>
* Activate the environment
```bash
conda activate citgen
```
<br>
* Install packages in the environment
```bash
pip install -r requirements.txt
```

Suppose your cuda version is cuda1XX, then in the terminal run the following command:
```bash
cp YOUR_CONDA_PATH/envs/citgen/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cuda1XX.so YOUR_CONDA_PATH/envs/citgen/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cpu.so
```


<s>**Note** If there are errors during installation, please try create environment using the environment.yml file: 
conda env create -f environment.yml
</s>


## Configure wandb 
a tool for monitoring training proress, similar to tensorboard
```bash
wandb init
```

## Configure Huggingface Account
You need a huggingface account to load model's checkpoint from the cloud
```bash
huggingface-cli login
```

## Download Models' checkpoints

### SciBERT-based citation intent classifier
```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="nianlong/scibert-citation-intent-classifier", 
                filename="model_batch_515.pt", 
                local_dir = "reward_model/citation_intent_classification/model/BertClassifier/5_5_0.05_0.01/")

```

## Download Dataset
```python
from datasets import load_dataset
import json
from tqdm import tqdm

for split, fname in [ ( "train", "train.jsonl"), ( "validation", "val.jsonl"),("test", "test.jsonl") ]:
    print(split, fname)

    dataset = load_dataset('nianlong/controllable-citation-generation', 
                            split = split,
                            num_proc = 16 
                         )
    with open("data/"+fname, "w") as f:
        for example in tqdm( dataset ):
            f.write( json.dumps( example ) + "\n" )
```


## Start Supervised Finetuning

### Decoder-only Models
#### Training the Decoder-only models. Supported models: LLaMa, Galactica, and GPT-Neo

```bash
NCCL_P2P_DISABLE=1 torchrun --rdzv-endpoint=localhost:29500 --nnodes 1  --nproc_per_node 2 train_sft.py --model_path facebook/galactica-125m --model_type galactica --output_dir sft_model/galactica-125m-peft --streaming 1 --max_steps 5000 --num_warmup_steps 500 --eval_freq 1000 --save_freq 1000 --log_freq 10 --batch_size 16 --gradient_accumulation_steps 8 --use_lora 1 --train_dataset_name data/train.jsonl --val_dataset_name data/val.jsonl --quantization int4 --learning_rate 1e-4 
```


Meaning of important parameters:
* model_path: the model path to the pretrained LM, e.g. facebook/galactica-125m
* model_type: the type of the model, e.g., "galactica", "gpt-neo", "llama". This is used to load the Tokenizer and LoRa config
* output_dir: path to save the checkpoints.
* streaming: whether to load the dataset as a file stream direct from disk. Setting this to 1 if your training corpus are very large, otherwise, set it to 0, then the whole corpus will be loaded to RAM and better shuffled.
* use_lora: whether to use LoRa to train the low-rank adapter. This is needed when finetune large LM such as Galactica-6.7B and LLaMa. Set it to 1 if you want to use LoRA. Here we set it to 1 for training Galactica-125m just for demonstration. When use_lora is 0, the saved checkpoints will be huggingface models, not the adapter models.
* quantization: quantization precision, can be int4 or int8, used only when use_lora is set to 1
* learning_rate: a higher learning rate such as 1e-4 is preferred when use_lora, otherwise set it to a smaller value such as 1e-5

More default parameters, such as lora configuration, can be found in the train.py file.

**Note: GPT-NEO is not supported by low-precision speedup in this script. So when training GPT-NEO please set use_lora to 0**


#### Merge the peft adapter with the original model to produce a huggingface model.
This is only needed if you fine-tuned the LM using LoRA, current only Galactica and LLaMa are supported.

For Galactica:

```bash
python convert_peft_to_hf.py --base_model_path facebook/galactica-125m --lora_model_path sft_model/galactica-125m-peft/checkpoint-5000 --save_model_path sft_model/galactica-125m-hf --model_type galactica
```

For LLaMa:

```bash
python convert_peft_to_hf.py --base_model_path huggyllama/llama-7b --lora_model_path sft_model/llama-7b-peft/checkpoint-5000 --save_model_path sft_model/llama-7b-hf --model_type llama
```


#### Convert huggingface model to Ctranslate model

For Galactica:
```bash
python convert_hf_to_ct2.py --model sft_model/galactica-125m-hf --output_dir sft_model/galactica-125m-ct2 --quantization int8_float16 --load_as_float16 --low_cpu_mem_usage --model_type galactica
```

For LLaMa:
```bash
python convert_hf_to_ct2.py --model sft_model/llama-7b-hf --output_dir sft_model/llama-7b-ct2 --quantization int8_float16 --load_as_float16 --low_cpu_mem_usage --model_type llama
```

### Encoder-Decoder Models. Supported models: BART

#### Training

```bash
NCCL_P2P_DISABLE=1 torchrun --rdzv-endpoint=localhost:29500 --nnodes 1 --nproc-per-node 2 train_sft_seq2seq_bart.py --model_path facebook/bart-base --output_dir sft_model/bart-base --streaming 1 --log_freq 1 --learning_rate 1e-5 --batch_size 64 --gradient_accumulation_steps 1 --train_dataset_name data/train.jsonl --val_dataset_name data/val.jsonl --max_steps 5000 --eval_freq 1000 --save_freq 1000 --num_warmup_steps 500
```

## Start PPO Finetuning
PPO Finetuning only support Decoder-only model: Galactica and LLaMa

### Training

For Galactica:

```bash
NCCL_P2P_DISABLE=1 accelerate launch --multi_gpu --num_machines 1  --num_processes 2 train_ppo.py --model_path sft_model/galactica-125m-hf/checkpoint-5000 --model_type galactica --output_dir ppo_model/galactica-125m-peft  --train_dataset_name data/train.jsonl --save_freq 10 --batch_size 256 --mini_batch_size 4 --gradient_accumulation_steps  1 --quantization int4
```
**Note: **
* When training using a single GPU, remove the  "--multi_gpu"  parameter
* --model_path point to the folder where the pytorch_model.bin is located. It can be sft_model/galactica-125m-hf or sft_model/galactica-125m-hf/checkpoint_XXX, depending on how the model is obtained, e.g., converted from peft model or saved during training


For LLaMa:
```bash
NCCL_P2P_DISABLE=1 accelerate launch --multi_gpu --num_machines 1  --num_processes 2 train_ppo.py --model_path sft_model/llama-7b-hf --model_type galactica --output_dir ppo_model/llama-7b-peft  --train_dataset_name data/train.jsonl --save_freq 10 --batch_size 256 --mini_batch_size 4 --gradient_accumulation_steps  1 --quantization int4
```

#### Merge the peft adapter with the original model to produce a huggingface model.
Same as the supervised finetuning stage.

**Note:**
When calling the convert_peft_to_hf.py, be aware that the parameter --base_model_path is no longer the pretrained language model, but the path to the supervised finetuned model! For example, for Galactica-125m it would be "sft_model/galactica-125m-hf", not "facebook/galactica-125m-hf"

#### Convert huggingface model to Ctranslate model
Same as the supervised finetuning stage.


#### Use the citation generator

```python
from generator import CitationGeneratorFast, CitationGenerator, BartCitationGenerator
import json
import evaluate
from tqdm import tqdm


model_path = "sft_model/galactica-125m-ct2"
# model_path = "sft_model/galactica-125m-hf"
# model_path = "ppo_model/galactica-125m-ct2"
# model_path = "sft_model/bart-base/checkpoint-5000"
# ...

model_architecture = "decoder"
# model_architecture = "encoder-decoder" # for BART model


if model_architecture == "decoder" and "bart" not in model_path:
    ## If it is ctranslated model
    if model_path.endswith("-ct2") or model_path.endswith("-ct2/"):
        cit_generator = CitationGeneratorFast( model_path )
    else:
        cit_generator = CitationGenerator( model_path )    
else:
    cit_generator = BartCitationGenerator( model_path )



corpus = [ json.loads(line) for line in open( "data/test.jsonl", "r") ]


example = corpus[50]

gen_cit_uncontrolled = cit_generator.generate(
            citing_paper_title = example["citing_paper_content"]["title"] ,
            citing_paper_abstract = example["citing_paper_content"]["abstract"],
            cited_paper_title =  example["cited_paper_content"]["title"],
            cited_paper_abstract = example["cited_paper_content"]["abstract"],
            text_before_citation = " ".join( example["text_before_citation"] ),
            
            num_beams = 1,
            
        )
print(gen_cit_uncontrolled["citation"])

gen_cit_with_intent = cit_generator.generate(
            citing_paper_title = example["citing_paper_content"]["title"] ,
            citing_paper_abstract = example["citing_paper_content"]["abstract"],
            cited_paper_title =  example["cited_paper_content"]["title"],
            cited_paper_abstract = example["cited_paper_content"]["abstract"],
            text_before_citation = " ".join( example["text_before_citation"] ),
            
            intent = example["citation_intent"],
            
            num_beams = 1,
            
        )

print(gen_cit_with_intent["citation"])
        
gen_cit_with_intent_and_keywords = cit_generator.generate(
            citing_paper_title = example["citing_paper_content"]["title"] ,
            citing_paper_abstract = example["citing_paper_content"]["abstract"],
            cited_paper_title =  example["cited_paper_content"]["title"],
            cited_paper_abstract = example["cited_paper_content"]["abstract"],
            text_before_citation = " ".join( example["text_before_citation"] ),
            
            intent = example["citation_intent"],
            keywords = "; ".join( example["keywords"] ),
            
            num_beams = 1,
            
        )

print(gen_cit_with_intent_and_keywords["citation"])


```


## Load fine-tuned citation generation models

We have pushed the finetuned citation generation huggingface models to huggingface hub. The model name are listed as below:

* nianlong/citgen-bart-base
* nianlong/citgen-bart-large
* nianlong/citgen-gpt-neo-125m-sft
* nianlong/citgen-gpt-neo-1.3b-sft
* nianlong/citgen-galactica-125m-sft
* nianlong/citgen-galactica-125m-ppo
* nianlong/citgen-galactica-1.3b-sft
* nianlong/citgen-galactica-6.7b-sft
* nianlong/citgen-galactica-6.7b-ppo
* nianlong/citgen-llama-7b-sft
* nianlong/citgen-llama-7b-ppo

**Note:** Before using the finetuned LLaMa-7b model (nianlong/citgen-llama-7b-sft and nianlong/citgen-llama-7b-ppo), please see Meta's [release](https://github.com/facebookresearch/llama)  and [request form](https://forms.gle/jk851eBVbX1m5TAv5).

We can directly use the model for citation generation, e.g.:

```python
from generator import CitationGeneratorFast, CitationGenerator, BartCitationGenerator
import json
import evaluate
from tqdm import tqdm

model_path = "nianlong/citgen-galactica-125m-sft"

model_architecture = "decoder"
# model_architecture = "encoder-decoder" # for BART model


if model_architecture == "decoder" and "bart" not in model_path:
    ## If it is ctranslated model
    if model_path.endswith("-ct2") or model_path.endswith("-ct2/"):
        cit_generator = CitationGeneratorFast( model_path )
    else:
        cit_generator = CitationGenerator( model_path )    
else:
    cit_generator = BartCitationGenerator( model_path )



corpus = [ json.loads(line) for line in open( "data/test.jsonl", "r") ]


example = corpus[50]

gen_cit_uncontrolled = cit_generator.generate(
            citing_paper_title = example["citing_paper_content"]["title"] ,
            citing_paper_abstract = example["citing_paper_content"]["abstract"],
            cited_paper_title =  example["cited_paper_content"]["title"],
            cited_paper_abstract = example["cited_paper_content"]["abstract"],
            text_before_citation = " ".join( example["text_before_citation"] ),
            
            num_beams = 1,
            
        )
print(gen_cit_uncontrolled["citation"])

gen_cit_with_intent = cit_generator.generate(
            citing_paper_title = example["citing_paper_content"]["title"] ,
            citing_paper_abstract = example["citing_paper_content"]["abstract"],
            cited_paper_title =  example["cited_paper_content"]["title"],
            cited_paper_abstract = example["cited_paper_content"]["abstract"],
            text_before_citation = " ".join( example["text_before_citation"] ),
            
            intent = example["citation_intent"],
            
            num_beams = 1,
            
        )

print(gen_cit_with_intent["citation"])
        
gen_cit_with_intent_and_keywords = cit_generator.generate(
            citing_paper_title = example["citing_paper_content"]["title"] ,
            citing_paper_abstract = example["citing_paper_content"]["abstract"],
            cited_paper_title =  example["cited_paper_content"]["title"],
            cited_paper_abstract = example["cited_paper_content"]["abstract"],
            text_before_citation = " ".join( example["text_before_citation"] ),
            
            intent = example["citation_intent"],
            keywords = "; ".join( example["keywords"] ),
            
            num_beams = 1,
            
        )

print(gen_cit_with_intent_and_keywords["citation"])

```

Or we can convert the Galactica or LLaMa model by Ctranslate2, using the script convert_hf_to_ct2.py, to speed up the inference.
