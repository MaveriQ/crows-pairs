import streamlit as st
import pandas as pd
from transformers import (
    BertTokenizer, BertForMaskedLM, 
    RobertaTokenizer, RobertaForMaskedLM,
    AlbertTokenizer, AlbertForMaskedLM
)

import numpy as np
import torch

@st.cache
def setup():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    model.eval()

    mask_token = tokenizer.mask_token
    mask_token_id = tokenizer.convert_tokens_to_ids(mask_token)
    log_softmax = torch.nn.LogSoftmax(dim=0)

    return {"model": model,
      "tokenizer": tokenizer,
      "mask_token": mask_token,
      "log_softmax": log_softmax,
      "mask_token_id": mask_token_id
    }

# @st.cache
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
    probs = np.exp(log_probs.detach().numpy())
    top_k_idx = (-hs.detach().numpy()).argsort()[:topk_select]
    top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_idx)


    return log_probs, probs ,top_k_tokens

st.set_page_config(layout="wide")
st.title('Sentence Analyzer')
sentence = st.text_input('Enter Sentence to be analyzed',value="He is a strong man")

lm = setup()
tokenizer = lm['tokenizer']

sentence_token_ids = tokenizer.encode(sentence, return_tensors='pt')
sentence_tokens = tokenizer.decode(sentence_token_ids[0]).split()

masked_tokens = st.multiselect('Select token IDs to mask (the order does not matter)',sentence_tokens,default=sentence_tokens)
topk_select = st.sidebar.number_input('Top K',min_value=0,max_value=20,value=5)

df = pd.DataFrame(index=sentence_tokens,columns=['log_prob', 'prob',f'top_{topk_select}_tokens'])

masked_token_index=[] # Find the indices in sentence_tokens that are masked_tokens
for i,y in enumerate(sentence_tokens):
    for x in masked_tokens:
        if x==y:
            masked_token_index.append(i)

maked_token_ids = tokenizer.convert_tokens_to_ids(masked_tokens)
sent_masked_token_ids = sentence_token_ids.clone().detach()

for i,token in enumerate(sentence_tokens):

    sent_masked_token_ids[0][i] = lm["mask_token_id"]
    score, probs, top_k_tokens = get_log_prob_unigram(sent_masked_token_ids, sentence_token_ids, i, lm)

    if i in masked_token_index:
        df.loc[sentence_tokens[i],'log_prob'] = score.item() # += score1.item()
        df.loc[sentence_tokens[i],'prob'] = probs
    else:
        df.loc[sentence_tokens[i],'log_prob'] = -np.inf
        df.loc[sentence_tokens[i],f'top_{topk_select}_tokens'] = top_k_tokens
        df.loc[sentence_tokens[i],'prob'] = 0.0

st.dataframe(df)