import streamlit as st

import pickle
import pandas as pd
from transformers import BertTokenizer
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import spacy
import pdb
import difflib

st.set_page_config(layout="wide")
st.title('CrowS-Pair Dataset Analysis')

@st.cache
def get_df():
    return pickle.load(open('./results/original.pkl','rb'))

@st.cache
def get_tokenizer():
    return BertTokenizer.from_pretrained('bert-base-uncased')

@st.cache
def get_viz(df,row):
    seq1 = tokenizer.encode(df.iloc[row].sent_more)
    seq2 = tokenizer.encode(df.iloc[row].sent_less)
    seq1 = [str(x) for x in seq1]
    seq2 = [str(x) for x in seq2]
    dataframes=[]

    matcher = difflib.SequenceMatcher(None, seq1, seq2)
    template1, template2 = [], []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        a=[tokenizer.ids_to_tokens[int(x)] for x in seq1[i1:i2]]
        b=[tokenizer.ids_to_tokens[int(x)] for x in seq2[j1:j2]]
        c=[df.iloc[row].sent_more_token_scores.get(x,-np.inf) for x in range(i1,i2)]
        d=[df.iloc[row].sent_less_token_scores.get(x,-np.inf) for x in range(j1,j2)]
        e=[0]*len(a)
        if tag!='equal':
            a=tokenizer.decode([int(x) for x in seq1[i1:i2]])
            b=tokenizer.decode([int(x) for x in seq2[j1:j2]])
            c=[-np.inf]
            d=[-np.inf]
            e=[1]*len(c)
        data = {'token_more':a,
                'token_less':b,
                'score_more':c,
                'score_less':d,
                'bias_tokens':e
               }
        dataframes.append(pd.DataFrame.from_dict(data))
        
    df2 = pd.concat(dataframes).reset_index(drop=True)
    df2 = df2.assign(prob_more=lambda x: np.exp(x.score_more),
                       prob_less=lambda x: np.exp(x.score_less),
                       prob_diff=lambda x: x.prob_more-x.prob_less,
                       prob_diff_scaled=lambda x: MinMaxScaler().fit_transform(x.prob_diff.values.reshape(-1, 1)))
    df2 = df2.iloc[1:-1,:] #removing CLS and SEP tokens
    return df2.drop('bias_tokens',axis=1,inplace=False)

@st.cache
def highlight_bias_token(s):
    '''
    highlight the bias token.
    '''
    val = ['background-color: yellow']*len(s) if s.score_more==-np.inf else ['']*len(s)
    return val

df = get_df()
tokenizer = get_tokenizer()

row = st.slider('Select Row',min_value=0,max_value=int(df.shape[0]))
precision = st.sidebar.slider('Numerical Precision',min_value=1,max_value=5,value=3)
model = st.sidebar.selectbox('Select Model',['bert','roberta'])
language = st.sidebar.selectbox('Select Language',['English','Chinese','Italian','German'])

# options = st.multiselect('Select tokens',tokenizer.encode('i am going to berlin'),default=tokenizer.encode('i am going to berlin'))

cmap_prob_diff = st.sidebar.selectbox('Colormap for prob_diff row',('green', 'rocket', 'lightblue', 'none'))
cmap_prob_diff_scaled = st.sidebar.selectbox('Colormap for prob_diff_scaled row',('green', 'rocket', 'lightblue', 'none'),index=2)

df_viz = get_viz(df,row)

cmaps = {
    'green' : sns.light_palette("green", as_cmap=True),
    'rocket' : sns.color_palette("rocket", as_cmap=True),
    'lightblue' : sns.color_palette("light:b", as_cmap=True)
}

# columns = st.multiselect('Select Columns to hide',df_viz.columns)

# df = df_viz.drop(columns,axis=1,inplace=False)

st.dataframe(df.T.style.apply(highlight_bias_token,axis=0)\
            .background_gradient(cmap=cmaps[cmap_prob_diff],subset=pd.IndexSlice['prob_diff',:],axis=1)\
            .background_gradient(cmap=cmaps[cmap_prob_diff_scaled],subset=(['prob_diff_scaled'],),axis=1).set_precision(precision),width=None)