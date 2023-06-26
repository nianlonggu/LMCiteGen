from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from accelerate import Accelerator
import torch
from tqdm import tqdm
import numpy as np

import os
import ctranslate2
import sentencepiece as spm

from peft import PeftModel

from nltk.tokenize import sent_tokenize, word_tokenize

from transformers import BartForConditionalGeneration


class CitationGeneratorBase:
    def __init__(self,):
        pass
        
    def get_context( self, citing_paper_title,
                           citing_paper_abstract,
                           cited_paper_title,
                           cited_paper_abstract,
                           text_before_citation
                   ):
        
        if isinstance( text_before_citation, list):
            text_before_citation = " ".join(text_before_citation)
        
        context = "\n\n".join(
            [
                "### Citing Paper Title:\n%s"%( citing_paper_title ),
                "### Citing Paper Abstract:\n%s"%( citing_paper_abstract ),
                "### Cited Paper Title:\n%s"%( cited_paper_title ),
                "### Cited Paper Abstract:\n%s"%( cited_paper_abstract ),
                "### Text Before Citation:\n%s"%( text_before_citation )
            ]
        )
        return context
    
    def extract_citation_sentence(self, text, min_num_words = 15 ):
        sentence_list = sent_tokenize(text)
        
        ## remove the last incomplete sentence
        if len(sentence_list) > 1 and not sentence_list[-1].endswith("."):
            sentence_list = sentence_list[:-1]
        
        cit_sen = ""
        for sen in sentence_list:
            if sen.endswith(" ."):
                sen = sen[:-2] + "."
            cit_sen += sen.strip() + " "
            if len(word_tokenize(cit_sen)) > min_num_words:
                break
        cit_sen = cit_sen.strip().replace("#REFR ,","#REFR,").replace("#REFR .","#REFR.")
        
        return cit_sen
    
    
class CitationGenerator(CitationGeneratorBase):
    def __init__(self, model_path, quantization = "int8"):
        super().__init__()
                
        tokenizer = AutoTokenizer.from_pretrained( model_path )
        tokenizer.padding_side="left"
        self.tokenizer = tokenizer
        
        bnb_config = BitsAndBytesConfig(        
            load_in_8bit= quantization == "int8",
            llm_int8_threshold=6.0,
            llm_int8_skip_modules=None,
            llm_int8_enable_fp32_cpu_offload=False,
            llm_int8_has_fp16_weight=False,
            load_in_4bit= quantization == "int4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb_config, device_map={"": Accelerator().process_index}
        )
        # if peft_model_path is not None:        
        #     model = PeftModel.from_pretrained(
        #         model,
        #         peft_model_path,
        #         torch_dtype=torch.bfloat16,
        #     )
        model.eval()
        self.model = model
        self.device = "cuda:%d"%( Accelerator().process_index )
        
    @torch.no_grad()
    def generate( self,  
                           citing_paper_title = "",
                           citing_paper_abstract = "",
                           cited_paper_title = "",
                           cited_paper_abstract = "",
                           text_before_citation = "",
                           intent = None, 
                           keywords = None,
                           
                           sampling_temperature=1.0,
                           sampling_topk=1,
                           num_beams = 2,
                 
                           only_attributes = False,
                ):
        prompt = self.get_context( citing_paper_title,
                           citing_paper_abstract,
                           cited_paper_title,
                           cited_paper_abstract,
                           text_before_citation )
        prompt = prompt + "\n\n### Citation Intent:\n"
        
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        if intent is None:
            out_ids = self.model.generate( input_ids=input_ids, 
                                           temperature=sampling_temperature,
                                           top_k = sampling_topk,
                                           num_beams = num_beams,
                                           max_new_tokens=3 )
            intent = self.tokenizer.decode(out_ids[0][-3:]).split("\n")[0]
            
        prompt = prompt + intent + "\n\n### Keywords:\n"
        
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        if keywords is None:
            out_ids = self.model.generate( input_ids=input_ids, 
                                           temperature=sampling_temperature,
                                           top_k = sampling_topk,
                                           num_beams = num_beams,
                                           max_new_tokens = 10 )
            keywords = self.tokenizer.decode(out_ids[0][-10:]).split("\n")[0]
        else:
            if isinstance(keywords, list):
                keywords = "; ".join(keywords)
                
        if only_attributes:
            return { 
                 "citation_intent":intent, 
                 "keywords":keywords, 
               }
                
        prompt = prompt + keywords + "\n\n### Citation:\n"
        
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        out_ids = self.model.generate(
                    input_ids=input_ids,
                    temperature=sampling_temperature,
                    top_k=sampling_topk,
                    num_beams = num_beams,
                    max_new_tokens=50,
                    min_new_tokens=15
        )
        
        citation = self.tokenizer.decode(out_ids[0][-60:], skip_special_tokens = True).split("Citation:\n")[-1].split("\n")[0]
        citation = self.extract_citation_sentence( citation, min_num_words = 15 )
        
        return { "citation_intent":intent, 
                 "keywords":keywords, 
                 "citation":citation
               }
    

class CitationGeneratorFast(CitationGeneratorBase):
    def __init__(self, model_path ):
        self.model = ctranslate2.Generator(model_path, device="cuda" if torch.cuda.is_available() else "cpu" )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
    def generate( self,  
                           citing_paper_title = "",
                           citing_paper_abstract = "",
                           cited_paper_title = "",
                           cited_paper_abstract = "",
                           text_before_citation = "",
                           intent = None, 
                           keywords = None,
                           
                           sampling_temperature=1.0,
                           sampling_topk=1,
                           num_beams = 2,
                 
                           only_attributes = False,
                ):
        prompt = self.get_context( citing_paper_title,
                           citing_paper_abstract,
                           cited_paper_title,
                           cited_paper_abstract,
                           text_before_citation )
        prompt = prompt + "\n\n### Citation Intent:\n"

        input_tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(prompt))                    
        if intent is None:
            step_results = self.model.generate_batch(
                [input_tokens],
                sampling_temperature=sampling_temperature,
                sampling_topk=sampling_topk,
                beam_size = num_beams,
                max_length=3,
                include_prompt_in_result = False
            )
            intent = self.tokenizer.decode( step_results[0].sequences_ids[0] ).split("\n")[0]
            
        prompt = prompt + intent + "\n\n### Keywords:\n"
        input_tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(prompt))    
        
        if keywords is None:
            step_results = self.model.generate_batch(
                [input_tokens],
                sampling_temperature=sampling_temperature,
                sampling_topk=sampling_topk,
                beam_size = num_beams,
                max_length=10,
                include_prompt_in_result = False
            )
            keywords = self.tokenizer.decode(step_results[0].sequences_ids[0]).split("\n")[0]
        else:
            if isinstance(keywords, list):
                keywords = "; ".join(keywords)
                
        if only_attributes:
            return { 
                 "citation_intent":intent, 
                 "keywords":keywords, 
               }
        
        prompt = prompt + keywords + "\n\n### Citation:\n"
        input_tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(prompt))
        
        step_results = self.model.generate_batch(
                [input_tokens],
                sampling_temperature=sampling_temperature,
                sampling_topk=sampling_topk,
                beam_size = num_beams,
                max_length=50,
                min_length=15,
                include_prompt_in_result = False
            )
        citation = self.tokenizer.decode(step_results[0].sequences_ids[0])
        citation = self.extract_citation_sentence( citation, min_num_words = 15 )
        
        return { "citation_intent":intent, 
                 "keywords":keywords, 
                 "citation":citation
               }
    


class BartCitationGenerator(CitationGeneratorBase):
    def __init__(self, model_path ):
        super().__init__()
                
        self.tokenizer =  AutoTokenizer.from_pretrained( model_path )
        model = BartForConditionalGeneration.from_pretrained(model_path).to( "cuda:%d"%( Accelerator().process_index ) )
        model.eval()
        self.model = model
        self.device = "cuda:%d"%( Accelerator().process_index )
        
    @torch.no_grad()
    def generate( self,  
                           citing_paper_title = "",
                           citing_paper_abstract = "",
                           cited_paper_title = "",
                           cited_paper_abstract = "",
                           text_before_citation = "",
                           intent = None, 
                           keywords = None,
                           
                           sampling_temperature=1.0,
                           sampling_topk=1,
                           num_beams = 2,
                 
                           only_attributes = False,
                ):
        encoder_input_text = self.get_context( citing_paper_title,
                           citing_paper_abstract,
                           cited_paper_title,
                           cited_paper_abstract,
                           text_before_citation )
        
        encoder_input_ids = self.tokenizer(encoder_input_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        
        prompt = "### Citation Intent:\n"
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[:,:-1].to(self.device)
        
        if intent is None:
            out_ids = self.model.generate( 
                                           input_ids = encoder_input_ids,
                                           decoder_input_ids=input_ids, 
                                           temperature=sampling_temperature,
                                           top_k = sampling_topk,
                                           num_beams = num_beams,
                                           max_new_tokens=3,
                                           min_new_tokens=3
            )
            
            intent = self.tokenizer.decode(out_ids[0][-3:]).split("\n")[0]
            
        prompt = prompt + intent + "\n\n### Keywords:\n"
        
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[:,:-1].to(self.device)
        if keywords is None:
            out_ids = self.model.generate( 
                                           input_ids = encoder_input_ids,
                                           decoder_input_ids=input_ids, 
                                           temperature=sampling_temperature,
                                           top_k = sampling_topk,
                                           num_beams = num_beams,
                                           max_new_tokens = 10,
                                           min_new_tokens = 10,
                                         )
            keywords = self.tokenizer.decode(out_ids[0][-10:]).split("\n")[0]
        else:
            if isinstance(keywords, list):
                keywords = "; ".join(keywords)
                
        if only_attributes:
            return { 
                 "citation_intent":intent, 
                 "keywords":keywords, 
               }
                
        prompt = prompt + keywords + "\n\n### Citation:\n"
        
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[:,:-1].to(self.device)
        out_ids = self.model.generate(
                    input_ids = encoder_input_ids,
                    decoder_input_ids=input_ids,
                    temperature=sampling_temperature,
                    top_k=sampling_topk,
                    num_beams = num_beams,
                    max_new_tokens=50,
                    min_new_tokens=15
        )
        
        citation = self.tokenizer.decode(out_ids[0][-60:], skip_special_tokens = True).split("Citation:\n")[-1].split("\n")[0]
        citation = self.extract_citation_sentence( citation, min_num_words = 15 )
        
        return { "citation_intent":intent, 
                 "keywords":keywords, 
                 "citation":citation
               }
    