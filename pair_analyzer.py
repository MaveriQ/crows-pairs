import streamlit as st

from transformers import (
    BertTokenizer, BertForMaskedLM, 
    RobertaTokenizer, RobertaForMaskedLM,
    AlbertTokenizer, AlbertForMaskedLM
)
import torch
from metric import get_span
import difflib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

st.set_page_config(layout="wide")
st.title('Pair Analyzer')

@st.cache
def setup(lm_model):
    if lm_model == "bert":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        uncased = True
    elif lm_model == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        model = RobertaForMaskedLM.from_pretrained('roberta-large')
        uncased = False
    elif lm_model == "albert":
        tokenizer = AlbertTokenizer.from_pretrained('albert-xxlarge-v2')
        model = AlbertForMaskedLM.from_pretrained('albert-xxlarge-v2')
        uncased = True

    mask_token = tokenizer.mask_token
    mask_token_id = tokenizer.convert_tokens_to_ids(mask_token)
    log_softmax = torch.nn.LogSoftmax(dim=0)

    return {"model": model,
      "tokenizer": tokenizer,
      "mask_token": mask_token,
      "log_softmax": log_softmax,
      "mask_token_id": mask_token_id
    }

sent1 = st.text_input('Enter First Sentence',value="He is strong")
sent2 = st.text_input('Enter Second Sentence',value="She is strong")
precision = st.sidebar.slider('Numerical Precision',min_value=1,max_value=5,value=3)
# lm_model = st.sidebar.selectbox('Select Model',['bert','roberta','albert'])

lm = setup(lm_model='bert')
tokenizer = lm['tokenizer']

sent1_token_ids = tokenizer.encode(sent1, return_tensors='pt')
sent2_token_ids = tokenizer.encode(sent2, return_tensors='pt')

def get_log_prob_unigram(masked_token_ids, token_ids, mask_idx, lm):
    """
    Given a sequence of token ids, with one masked token, return the log probability of the masked token.
    """
    
    model = lm["model"]
    tokenizer = lm["tokenizer"]
    log_softmax = lm["log_softmax"]
    mask_token = lm["mask_token"]

    # get model hidden states
    output = model(masked_token_ids)
    hidden_states = output[0].squeeze(0)
    mask_id = tokenizer.convert_tokens_to_ids(mask_token)

    # we only need log_prob for the MASK tokens
    assert masked_token_ids[0][mask_idx] == mask_id

    hs = hidden_states[mask_idx]
    target_id = token_ids[0][mask_idx]
    log_probs = log_softmax(hs)[target_id]

    return log_probs

def evaluate(sent1_token_ids,sent2_token_ids,lm):

    template1, template2 = get_span(sent1_token_ids[0], sent2_token_ids[0])

    sent1_log_probs = {}
    sent2_log_probs = {}
    total_masked_tokens = 0
    N = len(template1)

    # skipping CLS and SEP tokens, they'll never be masked
    for i in range(1, N-1):
        sent1_masked_token_ids = sent1_token_ids.clone().detach()
        sent2_masked_token_ids = sent2_token_ids.clone().detach()

        sent1_masked_token_ids[0][template1[i]] = lm["mask_token_id"]
        sent2_masked_token_ids[0][template2[i]] = lm["mask_token_id"]
        total_masked_tokens += 1

        score1 = get_log_prob_unigram(sent1_masked_token_ids, sent1_token_ids, template1[i], lm)
        score2 = get_log_prob_unigram(sent2_masked_token_ids, sent2_token_ids, template2[i], lm)

        sent1_log_probs[template1[i]] = score1.item() # += score1.item()
        sent2_log_probs[template2[i]] = score2.item() # += score2.item()

    pair = {
        'sent1_token_ids':sent1_token_ids,
        'sent2_token_ids':sent2_token_ids,
        'sent1_log_probs':sent1_log_probs,
        'sent2_log_probs':sent2_log_probs
    }

    return pair

def get_diff(pair):
    sent1_token_ids = pair['sent1_token_ids'][0]
    sent2_token_ids = pair['sent2_token_ids'][0]
    sent1_log_probs = pair['sent1_log_probs']
    sent2_log_probs = pair['sent2_log_probs']    
    
    seq1 = [str(x) for x in sent1_token_ids.tolist()]
    seq2 = [str(x) for x in sent2_token_ids.tolist()]
    dataframes=[]

    matcher = difflib.SequenceMatcher(None, seq1, seq2)
    template1, template2 = [], []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        a=[tokenizer.ids_to_tokens[int(x)] for x in seq1[i1:i2]]
        b=[tokenizer.ids_to_tokens[int(x)] for x in seq2[j1:j2]]
        c=[sent1_log_probs.get(x,-np.inf) for x in range(i1,i2)]
        d=[sent2_log_probs.get(x,-np.inf) for x in range(j1,j2)]
        e=[0]*len(a)
        if tag!='equal':
            a=tokenizer.decode([int(x) for x in seq1[i1:i2]])
            b=tokenizer.decode([int(x) for x in seq2[j1:j2]])
            c=[-np.inf]
            d=[-np.inf]
            e=[1]*len(c)
        data = {'token_sent1':a,
                'token_sent2':b,
                'score_sent1':c,
                'score_sent2':d,
                'bias_tokens':e
               }
        dataframes.append(pd.DataFrame.from_dict(data))
        
    df = pd.concat(dataframes).reset_index(drop=True)
    df = df.assign(prob_sent1=lambda x: np.exp(x.score_sent1),
                   prob_sent2=lambda x: np.exp(x.score_sent2),
                   prob_diff=lambda x: x.prob_sent1-x.prob_sent2,
                   prob_diff_scaled=lambda x: MinMaxScaler().fit_transform(x.prob_diff.values.reshape(-1, 1)))
    df = df.iloc[1:-1,:] #removing CLS and SEP tokens
    return df.drop('bias_tokens',axis=1,inplace=False)

@st.cache
def highlight_bias_token(s):
    '''
    highlight the bias token.
    '''
    val = ['background-color: yellow']*len(s) if s.score_sent1==-np.inf else ['']*len(s)
    return val

cmaps = {
    'green' : sns.light_palette("green", as_cmap=True),
    'rocket' : sns.color_palette("rocket", as_cmap=True),
    'lightblue' : sns.color_palette("light:b", as_cmap=True),
    'none' : 'seismic'
}

# https://matplotlib.org/stable/tutorials/colors/colormaps.html
cmap_prob_diff = st.sidebar.selectbox('Colormap for prob_diff row',('green', 'rocket', 'lightblue', 'none'))
cmap_prob_diff_scaled = st.sidebar.selectbox('Colormap for prob_diff_scaled row',('green', 'rocket', 'lightblue', 'none'),index=2)

pair = evaluate(sent1_token_ids,sent2_token_ids,lm)
df = get_diff(pair)

st.dataframe(df.T.style.apply(highlight_bias_token,axis=0)\
            .background_gradient(cmap=cmaps[cmap_prob_diff],subset=pd.IndexSlice['prob_diff',:],axis=1)\
            .background_gradient(cmap=cmaps[cmap_prob_diff_scaled],subset=(['prob_diff_scaled'],),axis=1).set_precision(precision),width=None)