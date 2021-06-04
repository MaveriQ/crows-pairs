import streamlit as st

import pickle
import pandas as pd
from transformers import BertTokenizer
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import spacy
import pdb

st.set_page_config(layout="wide")
st.title('My first app')
@st.cache
def setup():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    df = pickle.load(open('./results/original.pkl','rb'))
    return df,tokenizer

@st.cache
def get_viz(df,row=0):
    token_ids_more = tokenizer.encode(df.iloc[row].sent_more)
    token_ids_less = tokenizer.encode(df.iloc[row].sent_less)
    tokens_more = [tokenizer.ids_to_tokens[id] for id in token_ids_more]
    tokens_less = [tokenizer.ids_to_tokens[id] for id in token_ids_less]
    scores_more = df.iloc[row].sent_more_token_scores
    scores_less = df.iloc[row].sent_less_token_scores
    bias_tokens = []
    for a,b in zip(token_ids_more,token_ids_less):
        bias_tokens.append(a!=b)
    data = []
    for i in range(1,len(tokens_more)-1):
        record = (tokens_more[i],scores_more.get(i,-np.inf),tokens_less[i],scores_less.get(i,-np.inf),bias_tokens[i])
        data.append(record)
    df = pd.DataFrame.from_records(data,columns=['tokens_more','score_more','tokens_less','score_less','bias_tokens'])
    df = df.assign(prob_more=lambda x: np.exp(x.score_more),
                   prob_less=lambda x: np.exp(x.score_less),
                   prob_diff=lambda x: x.prob_more-x.prob_less,
                   prob_diff_scaled=lambda x: MinMaxScaler().fit_transform(x.prob_diff.values.reshape(-1, 1)))
    return df.drop('bias_tokens',axis=1,inplace=False)


def highlight_bias_token(s):
    '''
    highlight the bias token.
    '''
    val = ['background-color: yellow']*df.shape[1] if s.score_more==-np.inf else ['']*df.shape[1]
    return val

df,tokenizer = setup()

row = st.slider('Select Row',min_value=0,max_value=int(df.shape[0]))
precision = st.sidebar.slider('Numerical Precision',min_value=1,max_value=5,value=3)
df_viz = get_viz(df,row)

green = sns.light_palette("green", as_cmap=True)
rocket = sns.color_palette("rocket", as_cmap=True)
lightblue = sns.color_palette("light:b", as_cmap=True)

# st.dataframe(df_viz.style.apply(highlight_bias_token,axis=1)\
#             .background_gradient(cmap=lightblue,subset=['prob_diff'])\
#             .background_gradient(cmap=lightblue,subset=['prob_diff_scaled']).hide_columns(['bias_tokens']))



columns = st.multiselect('Select Columns to hide',df_viz.columns)

df = df_viz.drop(columns,axis=1,inplace=False)

st.dataframe(df.T.style.apply(highlight_bias_token,axis=0)\
            .background_gradient(cmap=lightblue,subset=pd.IndexSlice['prob_diff',:],axis=1)\
            .background_gradient(cmap=green,subset=(['prob_diff_scaled'],),axis=1).set_precision(precision),width=None)