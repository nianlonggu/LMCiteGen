from generator import CitationGeneratorFast, CitationGenerator, BartCitationGenerator
import json
import evaluate
from tqdm import tqdm
import os
import argparse
from transformers import logging
logging.set_verbosity_error()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,  )
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--save_path", type=str,  )
    parser.add_argument("--num_beams", type=int, default = 1 )
    parser.add_argument("--start", type=int, default = None )
    parser.add_argument("--size", type=int, default = None )
    parser.add_argument("--model_architecture", type=str, default = "decoder" ) 
    
    args = parser.parse_args()
    
    corpus = [ json.loads(line) for line in open( args.data_path, "r") ]
    
    if args.start is None:
        args.start = 0
        args.size = len(corpus)
    else:
        if args.size is None:
            args.size = len(corpus)
        args.save_path += "_%d"%( args.start )
        
    os.makedirs( os.path.dirname(args.save_path), exist_ok=True )
    
    if args.model_architecture == "decoder" and "bart" not in args.model_path:
    
        if args.model_path.endswith("-ct2"):
            cit_generator = CitationGeneratorFast( args.model_path )
        else:
            cit_generator = CitationGenerator( args.model_path )
        
    else:
        cit_generator = BartCitationGenerator( args.model_path )
        
        
    fw = open(args.save_path, "w")
    
    for example in tqdm( corpus[ args.start : args.start + args.size ] ):
        
        gen_cit_uncontrolled = cit_generator.generate(
            citing_paper_title = example["citing_paper_content"]["title"] ,
            citing_paper_abstract = example["citing_paper_content"]["abstract"],
            cited_paper_title =  example["cited_paper_content"]["title"],
            cited_paper_abstract = example["cited_paper_content"]["abstract"],
            text_before_citation = " ".join( example["text_before_citation"] ),
            
            num_beams = args.num_beams,
            
        )
        
        gen_cit_with_intent = cit_generator.generate(
            citing_paper_title = example["citing_paper_content"]["title"] ,
            citing_paper_abstract = example["citing_paper_content"]["abstract"],
            cited_paper_title =  example["cited_paper_content"]["title"],
            cited_paper_abstract = example["cited_paper_content"]["abstract"],
            text_before_citation = " ".join( example["text_before_citation"] ),
            
            intent = example["citation_intent"],
            
            num_beams = args.num_beams,
            
        )
        
        gen_cit_with_intent_and_keywords = cit_generator.generate(
            citing_paper_title = example["citing_paper_content"]["title"] ,
            citing_paper_abstract = example["citing_paper_content"]["abstract"],
            cited_paper_title =  example["cited_paper_content"]["title"],
            cited_paper_abstract = example["cited_paper_content"]["abstract"],
            text_before_citation = " ".join( example["text_before_citation"] ),
            
            intent = example["citation_intent"],
            keywords = "; ".join( example["keywords"] ),
            
            num_beams = args.num_beams,
            
        )
        
        example["generated_citations"]= [
            {
                "model":args.model_path,
                "generation":gen_cit_uncontrolled,
                "given_citation_intent":None,
                "given_keywords":None
            },
            {
                "model":args.model_path,
                "generation":gen_cit_with_intent,
                "given_citation_intent":example["citation_intent"],
                "given_keywords":None
            },
            {
                "model":args.model_path,
                "generation":gen_cit_with_intent_and_keywords,
                "given_citation_intent":example["citation_intent"],
                "given_keywords":example["keywords"]
            },
        ]  
        fw.write( json.dumps( example ) + "\n" )
    
    fw.close()

