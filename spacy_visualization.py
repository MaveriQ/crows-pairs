from spacy_streamlit.util import get_svg

import streamlit as st
from bert_nlp import nlp
import pickle
import pandas as pd
import transformers 
from spacy import displacy
import spacy
import numpy as np

spacy.tokens.Token.set_extension("log_prob", default=-np.inf,force=True)

nlp_en = spacy.load('en_core_web_lg')

bert = pickle.load(open('en_bert.pkl','rb'))

def get_displacy_viz_more(row):
    sent_more = nlp(bert.sent_more[row])
    words = []
    arcs = []

    for i,token in enumerate(sent_more):
        token._.log_prob = round(bert.sent_more_token_scores[row].get(i,-np.inf),5)

        words.append({'text':token.text,'tag':token._.log_prob})
        if token.i!=token.head.i:
            if token.i<token.head.i:
                direction = 'left'
                start = token.i
                end = token.head.i
            else:
                direction = 'right'
                start = token.head.i
                end = token.i
            arcs.append({"start": start, "end": end, "dir": direction, 'label':token.dep_})
    return {"words":words,"arcs":arcs}

def visualize_parser_more(row):
    key='more'
    cols = st.beta_columns(3)
    st.header("Visualization of sent_more")
    options = {
        "collapse_punct": cols[0].checkbox(
            "Collapse punct", value=True, key=f"{key}_parser_collapse_punct"
        ),
        "collapse_phrases": cols[1].checkbox(
            "Collapse phrases", key=f"{key}_parser_collapse_phrases"
        ),
        "compact": cols[2].checkbox("Compact mode", key=f"{key}_parser_compact"),
    }
    html = displacy.render(get_displacy_viz_more(row), options=options, style="dep",manual=True)
    html = html.replace("\n\n", "\n")
    st.write(get_svg(html), unsafe_allow_html=True)

def get_displacy_viz_less(row):
    sent_more = nlp(bert.sent_less[row])
    words = []
    arcs = []

    for i,token in enumerate(sent_more):
        token._.log_prob = round(bert.sent_less_token_scores[row].get(i,-np.inf),5)

        words.append({'text':token.text,'tag':token._.log_prob})
        if token.i!=token.head.i:
            if token.i<token.head.i:
                direction = 'left'
                start = token.i
                end = token.head.i
            else:
                direction = 'right'
                start = token.head.i
                end = token.i
            arcs.append({"start": start, "end": end, "dir": direction, 'label':token.dep_})
    return {"words":words,"arcs":arcs}

def visualize_parser_less(row):
    key='less'
    cols = st.beta_columns(3)
    st.header("Visualization of sent_less")
    options = {
        "collapse_punct": cols[0].checkbox(
            "Collapse punct", value=True, key=f"{key}_parser_collapse_punct"
        ),
        "collapse_phrases": cols[1].checkbox(
            "Collapse phrases", key=f"{key}_parser_collapse_phrases"
        ),
        "compact": cols[2].checkbox("Compact mode", key=f"{key}_parser_compact"),
    }
    html = displacy.render(get_displacy_viz_less(row), options=options, style="dep",manual=True)
    html = html.replace("\n\n", "\n")
    st.write(get_svg(html), unsafe_allow_html=True)

row = st.slider('Select Row',min_value=0,max_value=int(bert.shape[0]))
visualize_parser_more(row)
visualize_parser_less(row)