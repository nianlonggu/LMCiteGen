import json
import numpy as np
import time
from tqdm import tqdm
import requests
import re
import evaluate
import argparse


class ChatGPTCitationGenerator:
    def __init__(self, api_key, 
                       model = "gpt-3.5-turbo-0301", temperature = 0.1, 
                       api_address = 'https://api.openai.com/v1/chat/completions',
                       timeout = 15.0,
                       max_num_try = 5
                ):
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}',
        }
        self.model = model
        self.temperature = temperature
        self.api_address = api_address
        self.timeout = timeout
        self.max_num_try = max_num_try
        
    def get_query_controlled_mode(self, cited_title, cited_abstract,
                                   citing_title, citing_abstract,
                                   text_before_citation,
                                   citation_intent,
                                   keywords):
        query = """
The authors need to cite the reference paper:

    Title: %s 
    Abstract: %s
    
, in the manuscript with the global context:

    Title: %s
    Abstract: %s

, immediately following the provided local context:

    %s
    
Your task:
Please generate one citation sentence that cites the reference paper, seamlessly follows the local context, reflects the specified citation intent, and incorporates the specified keywords.

Requirements:
1. The generated citation sentence should reflect the citation intent: %s. Citation intents include:
    1) background: The citation provides background information or additional context about a relevant problem, concept, approach, or topic.
    2) method: The citation refers to the use of a specific method, tool, approach, or dataset from the reference paper.
    3) result: The citation compares or contrasts the results or findings of the manuscript with those in the reference paper.
2. The generated citation sentence should contain the specified keywords: %s. All the provided keywords should be used. If no keywords are specified, please infer one or two keywords by yourself and generate the citation sentence based on them.
3. Insert the citation marker "#REFR" at the position in the sentence where the reference paper should be cited. 
4. Put the citation marker "#REFR" correctly in the generated citation sentence. The citation marker should replace the entire in-text citation (e.g., authors and year of publication), should not be enclosed in any brackets, and should be placed within the sentence before the ending punctuation.

Please return only the generated citation sentence. 
        """%(
             cited_title,
             cited_abstract,
             citing_title,
             citing_abstract,
             text_before_citation,
             citation_intent,
             keywords
        )
        return query
    
    
    def get_query_uncontrolled_mode(self, cited_title, cited_abstract,
                                   citing_title, citing_abstract,
                                   text_before_citation
                                   ):
        query = """

The authors need to cite the reference paper:

    Title: %s 
    Abstract: %s
    
, in the manuscript with the global context:

    Title: %s
    Abstract: %s

, immediately following the provided local context:

    %s
    
Your task:
Please generate a citation sentence that cites the reference paper and seamlessly follows the local context. The citation sentence should implicitly reflect one of the following citation intents and incorporate relevant keywords:

1) Background: The citation provides background information or additional context about a relevant problem, concept, approach, or topic.
2) Method: The citation refers to the use of a specific method, tool, approach, or dataset from the reference paper.
3) Result: The citation compares or contrasts the results or findings of the manuscript with those in the reference paper.

Requirements:
1. Insert the citation marker "#REFR" at the position in the sentence where the reference paper should be cited. 
2. Put the citation marker "#REFR" correctly in the generated citation sentence. The citation marker should replace the entire in-text citation (e.g., authors and year of publication), should not be enclosed in any brackets, and should be placed within the sentence before the ending punctuation.

Please return only the generated citation sentence. 


        """%(
             cited_title,
             cited_abstract,
             citing_title,
             citing_abstract,
             text_before_citation
        )
        return query
        
    def controlled_generate( self, cited_title, cited_abstract,
                                   citing_title, citing_abstract,
                                   text_before_citation,
                                   citation_intent,
                                   keywords
                           ):
        
        data = {
          "model": self.model,
          "temperature":self.temperature,
          "max_tokens":2000,
          "messages": [
                {"role": "system", "content": "You are a scientific writing assistant. Your task is to generate citation sentences for a given manuscript, following detailed instructions. These instructions involve taking into account the context, desired citation intent, specific keywords in your responses."},
                {"role": "user", "content": self.get_query_controlled_mode( cited_title, cited_abstract,
                                   citing_title, citing_abstract,
                                   text_before_citation,
                                   citation_intent,
                                   keywords )},
            ]
        }
        
        generated_citation = None
        for try_id in range( self.max_num_try ):
            if try_id > 0:
                print("Retrying ... %d"%( try_id ))
            try:
                response = requests.post(self.api_address, headers=self.headers, data=json.dumps(data), timeout=self.timeout)
                generated_citation = response.json()["choices"][0]["message"]["content"]
                break
            except:
                continue
                
        return generated_citation
    
    def uncontrolled_generate( self, cited_title, cited_abstract,
                                   citing_title, citing_abstract,
                                   text_before_citation,
                           ):
        
        data = {
          "model": self.model,
          "temperature":self.temperature,
          "max_tokens":2000,
          "messages": [
                {"role": "system", "content": "You are a scientific writing assistant. Your task is to infer the citation intent and relevant keywords based on the provided context, and generate a citation sentence for a given manuscript. The citation sentence should seamlessly follow the local context, reflect the inferred citation intent, and incorporate the inferred keywords."},
                {"role": "user", "content": self.get_query_uncontrolled_mode( cited_title, cited_abstract,
                                   citing_title, citing_abstract,
                                   text_before_citation )},
            ]
        }
        
        generated_citation = None
        for try_id in range( self.max_num_try ):
            if try_id > 0:
                print("Retrying ... %d"%( try_id ))
            try:
                response = requests.post(self.api_address, headers=self.headers, data=json.dumps(data), timeout=self.timeout)
                generated_citation = response.json()["choices"][0]["message"]["content"]
                break
            except:
                continue
                
        return generated_citation
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str )
    parser.add_argument("--data_path", type=str ) 
    parser.add_argument("--save_path", type=str )
    parser.add_argument("--start", type=int, default = None )
    parser.add_argument("--size", type=int, default = None )
    
    args = parser.parse_args()
        
    chatgpt_generator = ChatGPTCitationGenerator( args.api_key )
    corpus = [ json.loads(line) for line in open( args.data_path, "r") ]
    
    if args.start is None:
        args.start = 0
        args.size = len(corpus)
    else:
        assert args.size is not None
        args.save_path = args.save_path + "_%d"%(args.start)
        
    fw = open(args.save_path, "w")

    for example in tqdm(corpus[ args.start : args.start + args.size ]):
        
        gen_cit_uncontrolled = chatgpt_generator.uncontrolled_generate( 
             example["cited_paper_content"]["title"],
             example["cited_paper_content"]["abstract"],
             example["citing_paper_content"]["title"],
             example["citing_paper_content"]["abstract"],
             " ".join(example["text_before_citation"]),
        )
        
        gen_cit_with_intent = chatgpt_generator.controlled_generate( 
             example["cited_paper_content"]["title"],
             example["cited_paper_content"]["abstract"],
             example["citing_paper_content"]["title"],
             example["citing_paper_content"]["abstract"],
             " ".join(example["text_before_citation"]),
            
             example["citation_intent"],
             ""
        )
        
        gen_cit_with_intent_and_keywords = chatgpt_generator.controlled_generate( 
             example["cited_paper_content"]["title"],
             example["cited_paper_content"]["abstract"],
             example["citing_paper_content"]["title"],
             example["citing_paper_content"]["abstract"],
             " ".join(example["text_before_citation"]),
            
             example["citation_intent"],
             "; ".join( example["keywords"] ),
        )
        

        example["generated_citations"]= [
            {
                "model":chatgpt_generator.model,
                "text":gen_cit_uncontrolled,
                "given_citation_intent":None,
                "given_keywords":None
            },
            {
                "model":chatgpt_generator.model,
                "text":gen_cit_with_intent,
                "given_citation_intent":example["citation_intent"],
                "given_keywords":None
            },
            {
                "model":chatgpt_generator.model,
                "text":gen_cit_with_intent_and_keywords,
                "given_citation_intent":example["citation_intent"],
                "given_keywords":example["keywords"]
            },
            
            
        ]  
        
        fw.write( json.dumps( example ) + "\n" )
        
        
    fw.close()        