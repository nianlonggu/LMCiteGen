from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class IntentClassifier( nn.Module ):
    def __init__(self, initial_model_path, vocab_size = None, hidden_size = 768,
                    ln_hidden_size = 32,
                    num_intents = 3,
                    num_sections = 5,
                    num_worthiness = 2,
                    
                ):
        super().__init__()
        self.model = AutoModel.from_pretrained(initial_model_path)
        if vocab_size is not None:
            self.model.resize_token_embeddings( vocab_size )
            
        self.ln_hidden_intent = nn.Linear( hidden_size, ln_hidden_size )
        self.ln_out_intent = nn.Linear( ln_hidden_size, num_intents )
        
        self.ln_hidden_section = nn.Linear( hidden_size, ln_hidden_size )
        self.ln_out_section = nn.Linear( ln_hidden_size, num_sections )
        
        self.ln_hidden_worthiness = nn.Linear( hidden_size, ln_hidden_size )
        self.ln_out_worthiness = nn.Linear( ln_hidden_size, num_worthiness )
        
        self.cr_loss_cal = nn.CrossEntropyLoss(reduction="none")
        
        self.training = False
        
    def forward( self, input_ids, token_type_ids, attention_mask,
                    intent_labels = None, intent_mask = None,
                    section_labels = None, section_mask = None,
                    worthiness_labels = None, worthiness_mask = None,
                    lambda_section = 0.1, lambda_worthiness = 0.05,
                    dropout = 0.2
               ):
        
        model_out = self.model( input_ids = input_ids, 
                                token_type_ids = token_type_ids,
                                attention_mask = attention_mask
                              )
        last_hidden_state = model_out.last_hidden_state
        cls_hidden_state = last_hidden_state[:,0,:]
        
        logits_intent = self.ln_out_intent( F.dropout( F.relu(self.ln_hidden_intent( cls_hidden_state )), 
                                                       p = dropout if self.training else 0.0 )
                                          )
        logits_section = self.ln_out_section( F.dropout( F.relu(self.ln_hidden_section( cls_hidden_state )), 
                                                       p = dropout if self.training else 0.0 )
                                          )
        logits_worthiness = self.ln_out_worthiness( F.dropout( F.relu(self.ln_hidden_worthiness( cls_hidden_state )), 
                                                       p = dropout if self.training else 0.0 )
                                          )
        
        loss = None
        
        if intent_labels is not None:
            loss_intent = self.cr_loss_cal( logits_intent, intent_labels )
            if intent_mask is not None:
                loss_intent =  (loss_intent * intent_mask).sum()/( intent_mask.sum() + 1e-9 )
            else:
                loss_intent = loss_intent.mean()
            
            if loss is None:
                loss = loss_intent
            else:
                loss = loss + loss_intent

        if section_labels is not None:
            loss_section = self.cr_loss_cal( logits_section, section_labels )
            if section_mask is not None:
                loss_section = ( loss_section * section_mask ).sum() / ( section_mask.sum() + 1e-9 )
            else:
                loss_section = loss_section.mean()
            loss_section = lambda_section * loss_section
            
            if loss is None:
                loss = loss_section
            else:
                loss = loss + loss_section
        
        if worthiness_labels is not None:
            loss_worthiness = self.cr_loss_cal( logits_worthiness, worthiness_labels )
            if worthiness_mask is not None:
                loss_worthiness = ( loss_worthiness * worthiness_mask ).sum() / ( worthiness_mask.sum() + 1e-9 )
            else:
                loss_worthiness = loss_worthiness.mean()
            loss_worthiness = lambda_worthiness * loss_worthiness
            
            if loss is None:
                loss = loss_worthiness
            else:
                loss = loss + loss_worthiness
            
        if loss is not None:
            return loss
        else:
            return logits_intent
        

class IntentClassifierWrapper:
    def __init__(self, model, tokenizer ):
        self.model = model
        self.device = list(model.parameters())[0].device
        self.tokenizer = tokenizer
        self.intent_label_mapper = {
            "background":0,
            "method":1,
            "result":2
        }
        self.intent_id_mapper = {
            0:"background",
            1:"method",
            2:"result"
        }
        
    def classify_intent(self, citation_text, max_input_length = 100, return_intent_in_text = True ):
        encoded_citation_text = self.tokenizer( citation_text,  
                                                max_length = max_input_length, 
                                                truncation = True,
                                                return_tensors = "pt"
                                              )
        input_ids = encoded_citation_text.input_ids.to(self.device)
        token_type_ids = encoded_citation_text.token_type_ids.to(self.device)
        attention_mask = encoded_citation_text.attention_mask.to(self.device)
        
        with torch.no_grad():
            logits_intent = self.model( input_ids, token_type_ids, attention_mask )
                        
            predicted_intent_id = torch.argmax( logits_intent[0] ).item()
            
        if return_intent_in_text:
            return self.intent_id_mapper[predicted_intent_id]
        else:
            return predicted_intent_id
        
    def get_intent_scores(self, citation_text, max_input_length = 100 ):
        encoded_citation_text = self.tokenizer( citation_text,  
                                                max_length = max_input_length, 
                                                truncation = True,
                                                return_tensors = "pt"
                                              )
        input_ids = encoded_citation_text.input_ids.to(self.device)
        token_type_ids = encoded_citation_text.token_type_ids.to(self.device)
        attention_mask = encoded_citation_text.attention_mask.to(self.device)
        
        with torch.no_grad():
            logits_intent = self.model( input_ids, token_type_ids, attention_mask )
            scores = logits_intent[0].softmax(0).detach().cpu().numpy()
            
        intent_scores = {}
        for pos in range(len(scores)):
            intent_scores[ self.intent_id_mapper[pos] ] = scores[pos]
        return intent_scores
        
        
class CitationIntentClassifier:
    def __init__(self, model_path, initial_model_path, gpu = None, hidden_size = 768, 
                 ln_hidden_size = 32, num_intents = 3, num_sections = 5, num_worthiness = 2 ):
        model = IntentClassifier( 
                    initial_model_path = initial_model_path, 
                    hidden_size = hidden_size,
                    ln_hidden_size = ln_hidden_size,
                    num_intents = num_intents,
                    num_sections = num_sections,
                    num_worthiness = num_worthiness
        )
        ckpt = torch.load( model_path,  map_location=torch.device('cpu') )
        model.load_state_dict( ckpt["model"] )        
        if gpu is not None and gpu < torch.cuda.device_count():
            device = torch.device("cuda:%d"%(gpu))
        else:
            device = torch.device("cpu")
        model = model.to(device)

        tokenizer = AutoTokenizer.from_pretrained( initial_model_path )
        
        self.intent_classifier_wrapper =  IntentClassifierWrapper( model, tokenizer )
        self.intent_label_mapper = self.intent_classifier_wrapper.intent_label_mapper
        self.intent_id_mapper = self.intent_classifier_wrapper.intent_id_mapper
        
        
    def classify_intent( self, citation_text, max_input_length = 100, return_intent_in_text = True ):
        return self.intent_classifier_wrapper.classify_intent( citation_text, max_input_length, return_intent_in_text )
    
    def get_intent_scores( self, citation_text, max_input_length = 100 ):
        return self.intent_classifier_wrapper.get_intent_scores( citation_text, max_input_length )