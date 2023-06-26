import random
import json
import os
import streamlit as st
from pathlib import Path
import base64
import uuid
import numpy as np


def display_global_context(title, abstract):
    st.markdown(f"### {title}")
    st.markdown(f"### {title}")
    st.markdown(f"### Abstract")
    st.markdown(f"{abstract}")


def get_data():
    corpus = [ json.loads(line) for line in open( st.session_state["data_file_path"]  ) ]
    
    found = False
    for idx in np.random.choice(len(corpus),len(corpus), replace = False ):
        if "user_preference" not in corpus[idx]:
            found = True
            break
    if not found:
        st.success("All %d examples have been evaluated! Thanks!"%(len(corpus)))
    else:    
        data = corpus[idx]
        np.random.shuffle( data["generated_citations"] )
        
        st.session_state["data"] = data
        st.session_state["idx"] = idx

def save_data():
    corpus = [ json.loads(line) for line in open( st.session_state["data_file_path"]  ) ]
    assert st.session_state["idx"] is not None and "user_preference" in st.session_state["data"]
    
    corpus[ st.session_state["idx"] ] = st.session_state["data"]
    with open( st.session_state["data_file_path"], "w" ) as f:
        for item in corpus:
            f.write( json.dumps( item ) + "\n" )
    
    
    
def display_data():
    data = st.session_state["data"]
        
    st.subheader("Cited Paper:")
    st.write("**Title:** "+ data['cited_paper_content']['title'] )
    st.write("**Abstract:** " + data['cited_paper_content']['abstract'])
    
    st.subheader("Manuscript:")
    st.write("**Title:** " + data['citing_paper_content']['title'] )
    st.write("**Abstract:** " + data['citing_paper_content']['abstract'] )
    st.write("**Text Before Citation:**" )
    st.write(" ".join(data['text_before_citation']))
    
    st.subheader("Citation Attributes:")
    st.write("**Citation Intent:** " + data['citation_intent'] )
    st.write("**Keywords:** "+"; ".join(data['keywords']))
    
    st.subheader("Ground-truth Citation Sentence:")
    st.write( data['citation'] )
    
    st.subheader("Generated Citation Sentences:")
    st.write("**Citation Sentence A:** ")
    st.write(data["generated_citations"][0]["generated_citation"])
    st.write("**Citation Sentence B:** ")
    st.write(data["generated_citations"][1]["generated_citation"])
    
    st.header("Evaluation:")
    options = ["A", "B", "No preference"]
    
    option_mapper = {
        "A":0,
        "B":1,
        "No preference":None
    }
    
    evaluation_criteria = {
       "IAS":"**Intent Alignment:** Which generated citation sentence better aligns with the specified intent attribute?",
       "KR":"**Keyword Recall:** Which generated citation sentence better incorporates the specified keywords attribute?",
       "FS":"**Fluency:** Which generated citation sentence is more grammatically correct and natural (fluent)?",
       "Similarity":"**Similarity:** Which generated citation sentence is more similar to the original (ground truth) citation sentence?"
    }
    user_preferences = {}

    for criterion in evaluation_criteria:
        user_preferences[criterion] = st.radio(evaluation_criteria[criterion], options, horizontal=True,
                                               key= str(st.session_state["run_id"]) + criterion
                                              )


    col1, col2 = st.columns([1,1])
    with col1:
        # Submit button
        if st.button('Submit'):
            st.session_state["count"] += 1
            data["user_preference"] = { metric:option_mapper[ user_preferences[metric] ] for metric in user_preferences  }
        
            save_data()
            get_data()
        
            st.session_state["run_id"] += 1
            st.experimental_rerun()
    
    with col2:
        # Skip button
        if st.button('Skip'):
            get_data()
            st.session_state["run_id"] += 1
            st.experimental_rerun()

    if st.session_state["count"] > 0:
        st.success("You have evaluated %d examples!"%(st.session_state["count"]))

def main():

    # Initial page config
    st.set_page_config(
         page_title='Controlled Citation Generation Evaluation',
        layout="wide"
    )
    
    if "data_file_path" not in st.session_state:
        st.session_state["data_file_path"] = "controlled_generation_comparison_results_galactica6.7B-PPO_ChatGPT.jsonl"
    if "data" not in st.session_state:
        st.session_state["data"] = None
    if "idx" not in st.session_state:
        st.session_state["idx"] = None
    if "count" not in st.session_state:
        st.session_state["count"] = 0
    if "run_id" not in st.session_state:
        st.session_state["run_id"] = 0
        
        
    st.title("Controlled Citation Generation Evaluation")
    
    if st.session_state["data"] is None:
        get_data()
        
    if st.session_state["data"] is not None:
        display_data()

    return None

if __name__ == '__main__':
    main()