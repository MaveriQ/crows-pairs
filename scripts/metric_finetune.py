# To run:
# python metric.py --input1 [path_file1] --input2 [path_file2] --metric [metric] --lm_model [model]
# input1 and input2 should have the same number of sentences
# the format is (sent_id sentences), separated by a white space
# metric: options are {baseline, mask-random, mask-predict}
# model: options are {bert, roberta, albert}

import os
import json
import math
import torch
import argparse
import difflib
import logging
import numpy as np
import pandas as pd
import random

from sklearn.model_selection import KFold
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm

from transformers import BertTokenizer, BertForMaskedLM
from transformers import AlbertTokenizer, AlbertForMaskedLM
from transformers import RobertaTokenizer, RobertaForMaskedLM
from transformers import AdamW, get_linear_schedule_with_warmup
from collections import defaultdict


def read_data(input_file1, input_file2):
    """
    Load data into panda DataFrame format.
    Each file should have format (separated by a white-space):
    sent_id sentence
    """
    
    df_data = pd.DataFrame(columns=['id', 'sent1', 'sent2'])
    
    data1 = [x.strip().split() for x in open(input_file1, 'r').readlines()]
    data2 = [x.strip().split() for x in open(input_file2, 'r').readlines()]

    assert len(data1) == len(data2)

    for sent1, sent2 in zip(data1, data2):
        sent_id = 'sent_' + sent1[0]

        df_item = {'id': sent_id,
                   'sent1': ' '.join(sent1[1:]),
                   'sent2': ' '.join(sent2[1:])}

        df_data = df_data.append(df_item, ignore_index=True)

    return df_data


def compute_log_prob(masked_token_ids, token_ids, lm):
    
    model = lm["model"]
    tokenizer = lm["tokenizer"]
    log_softmax = lm["log_softmax"]
    mask_token = lm["mask_token"]
    uncased = lm["uncased"]

    if torch.cuda.is_available():
        masked_token_ids = masked_token_ids.to('cuda')
        token_ids = token_ids.to('cuda')

    with torch.no_grad():
        # get model hidden states
        output = model(masked_token_ids)
        hidden_states = output[0].squeeze(0)

    log_probs = []
    sum_log_probs = 0.
    mask_id = tokenizer.convert_tokens_to_ids(mask_token)

    # we only need log_prob for the MASK tokens
    for i, token_id in enumerate(masked_token_ids[0]):
        if token_id.item() == mask_id:
            hs = hidden_states[i]
            target_id = token_ids[0][i]
            log_probs.append((i, log_softmax(hs)[target_id].item()))
            sum_log_probs += log_softmax(hs)[target_id].item()

    return log_probs, sum_log_probs


def get_span(seq1, seq2):

    seq1 = [str(x) for x in seq1.tolist()]
    seq2 = [str(x) for x in seq2.tolist()]

    matcher = difflib.SequenceMatcher(None, seq1, seq2)
    template1, template2 = [], []
    for op in matcher.get_opcodes():
        # each op is a list of tuple: 
        # (operation, pro_idx_start, pro_idx_end, anti_idx_start, anti_idx_end)
        # possible operation: replace, insert, equal
        # https://docs.python.org/3/library/difflib.html
        if op[0] == 'equal':
            template1 += [x for x in range(op[1], op[2], 1)]
            template2 += [x for x in range(op[3], op[4], 1)]

    return template1, template2


def mask_random(batch, lm, T=25):
    """
    Score each sentence using mask-random algorithm, following BERT masking algorithm.
    For each iteration, we randomly masked 15% of subword tokens output by model's tokenizer.
    T: number of iterations
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    N = batch[0].to(device)
    sent1 = batch[1].to(device)
    sent2 = batch[2].to(device)
    template1 = batch[3].to(device)
    template2 = batch[4].to(device)
    mask_id = batch[5].to(device)

    sent1_token_ids = lm['tokenizer'].encode(sent1, return_tensors = 'pt')
    sent2_token_ids = lm['tokenizer'].encode(sent2, return_tensors = 'pt')

    sent1_masked_token_ids = sent1_token_ids.clone().detach()
    sent2_masked_token_ids = sent2_token_ids.clone().detach()

    mask_prob = 0.15
    total_masked_tokens = 0
    
    sent1_log_probs = 0.
    sent2_log_probs = 0.
    num_masked_tokens = max(1, math.ceil(mask_prob * N))
    for t in range(T):
        masked_idx = np.random.choice(N, num_masked_tokens, replace=False)
        
        for idx in masked_idx:
            sent1_idx = template1[idx]
            sent2_idx = template2[idx]
            sent1_masked_token_ids[0][sent1_idx] = mask_id
            sent2_masked_token_ids[0][sent2_idx] = mask_id
            total_masked_tokens += 1

        _, score1 = compute_log_prob(sent1_masked_token_ids, sent1_token_ids, lm)
        _, score2 = compute_log_prob(sent2_masked_token_ids, sent2_token_ids, lm)
        sent1_log_probs += score1
        sent2_log_probs += score2

    return (sent1_log_probs-sent2_log_probs)**2


def mask_predict(batch, lm, T=10):
    """
    Score each sentence using mask-predict algorithm.
    For each iteration, we unmask n words until all the words are unmasked.
    T: number of iterations
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    N = batch[0].to(device)
    sent1 = batch[1].to(device)
    sent2 = batch[2].to(device)
    template1 = batch[3].to(device)
    template2 = batch[4].to(device)
    mask_id = batch[5].to(device)

    sent1_token_ids = lm['tokenizer'].encode(sent1, return_tensors = 'pt')
    sent2_token_ids = lm['tokenizer'].encode(sent2, return_tensors = 'pt')

    sent1_masked_token_ids = sent1_token_ids.clone().detach()
    sent2_masked_token_ids = sent2_token_ids.clone().detach()

    total_unmasked_tokens = 0
    sent1_log_probs = 0.
    sent2_log_probs = 0.
    log_probs1, log_probs2 = [], []
    for t in range(T):
        num_unmasked_tokens = int(N - (N * ((T - t) / T)))
        masked_idx = np.random.choice(N, num_unmasked_tokens, replace=False)

        if t == 0:
            # mask all tokens except the changing words
            for idx1, idx2 in zip(template1, template2):
                sent1_masked_token_ids[0][idx1] = mask_id
                sent2_masked_token_ids[0][idx2] = mask_id
        else:
            # sort log prob of tokens
            sorted_log_probs1 = sorted(log_probs1, key=lambda x: x[1], reverse=True)[:num_unmasked_tokens]
            sorted_log_probs2 = sorted(log_probs2, key=lambda x: x[1], reverse=True)[:num_unmasked_tokens]

            # get index of the token id that has the highest log prob
            for (idx1, _), (idx2, _) in zip(sorted_log_probs1, sorted_log_probs2):
                # make sure it is masked before
                assert sent1_masked_token_ids[0][idx1].item() == mask_id
                assert sent2_masked_token_ids[0][idx2].item() == mask_id

                # unmask the token 
                sent1_masked_token_ids[0][idx1] = sent1_token_ids[0][idx1]
                sent2_masked_token_ids[0][idx2] = sent2_token_ids[0][idx2]

                total_unmasked_tokens += 1

        log_probs1, score1 = compute_log_prob(sent1_masked_token_ids, sent1_token_ids, lm)
        log_probs2, score2 = compute_log_prob(sent2_masked_token_ids, sent2_token_ids, lm)

        sent1_log_probs += score1
        sent2_log_probs += score2

        # stop if all tokens already unmasked
        if total_unmasked_tokens == N:
            break

    return (sent1_log_probs-sent2_log_probs)**2

def get_lm(lm_model):
    model = None
    if alm_model == "bert":
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

    model.eval()
    if torch.cuda.is_available():
        model.to('cuda')

    mask_token = tokenizer.mask_token
    log_softmax = torch.nn.LogSoftmax(dim=0)

    vocab = tokenizer.get_vocab()
    with open(lm_model + ".vocab", "w") as f:
        f.write(json.dumps(vocab))

    return {"model": model,
      "tokenizer": tokenizer,
      "mask_token": mask_token,
      "log_softmax": log_softmax,
      "uncased": uncased
    }

def get_dataloader(train_df, tokenizer, uncased, mask_token):
    N_list = []
    sent1_list = []
    sent2_list = []
    template1_list = []
    template2_list = []
    mask_id_list = []

    for index, row in train_df.iterrows():
        sent1 = row['sent1']
        sent2 = row['sent2']
        if uncased:
            sent1 = sent1.lower()
            sent2 = sent2.lower()

        sent1_token_ids = tokenizer.encode(sent1, return_tensors = 'pt')
        sent2_token_ids = tokenizer.encode(sent2, return_tensors = 'pt')

        template1, template2 = get_span(sent1_token_ids[0], sent2_token_ids[0])
        mask_id = tokenizer.convert_tokens_to_ids(mask_token)
        N = len(template1)
        
        N_list.append(N)
        sent1_list.append(sent1)
        sent2_list.append(sent2)
        template1_list.append(template1)
        template2_list.append(template2)
        mask_id_list.append(mask_id)

    # Convert the lists into tensors.
    N = torch.cat(N_list, dim=0)
    sent1 = torch.cat(sent1_list, dim=0)
    sent2 = torch.cat(sent2_list, dim=0)
    template1 = torch.cat(template1_list, dim=0)
    template2 = torch.cat(template2_list, dim=0)
    mask_id = torch.cat(mask_id_list, dim=0)

    train_dataset = TensorDataset(N, sent1, sent2, template1, template2, mask_id)
    train_dataloader = DataLoader(
        train_dataset,
        sampler = RandomSampler(train_dataset),
        batch_size = 16
    )
    return train_dataloader


def evaluate(args):

    print("Evaluating:")
    print("Input1:", args.input1)
    print("Input2:", args.input2)
    print("Metric:", args.metric)
    print("Model:", args.lm_model)
    print("=" * 100)

    logging.basicConfig(level=logging.INFO)

    df_data = read_data(args.input1, args.input2)

    metric = baseline
    if args.metric == "mask-predict":
        metric = mask_predict
    elif args.metric == "mask-random":
        metric = mask_random

    kf = KFold(n_splits=6, shuffle=True)
    fold = 0
    for train_index, test_index in kf.split(df_data):
        print('FOLD ' + str(fold))

        train_pairs = df_data.iloc[train_index]
        test_pairs =  df_data.iloc[test_index]

        lm = get_lm(args.lm_model)

        train_dataloader = get_dataloader(train_pairs, lm['tokenizer'], lm['uncased'], lm['mask_token'])
        test_dataloader = get_dataloader(test_pairs, lm['tokenizer'], lm['uncased'], lm['mask_token'])

        optimizer = AdamW(lm['model'].parameters(), lr = 2e-5, eps = 1e-8)
        epochs = 2
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps = len(train_dataloader) * epochs)

        for epoch_i in range(0, epochs):
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            total_train_loss = 0
            lm['model'].train()
            for step, batch in enumerate(train_dataloader):
                lm['model'].zero_grad()
                loss = metric(batch, lm)
                total_train_loss += loss.item()
                loss.backward()
                clip_grad_norm(lm['model'].parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            avg_train_loss = total_train_loss / len(train_dataloader)
            print("  Average training loss: {0:.2f}".format(avg_train_loss))

            lm['model'].eval()
            total_eval_loss = 0
            nb_eval_steps = 0
            for batch in test_dataloader:
                with torch.no_grad():
                    loss = metric(batch, lm)
                    total_eval_loss += loss.item()
            avg_val_loss = total_eval_loss / len(test_dataloader)
            print("  Validation Loss: {0:.2f}".format(avg_val_loss))

    fold += 1



parser = argparse.ArgumentParser()
parser.add_argument("--input1", type=str, help="path to input file 1")
parser.add_argument("--input2", type=str, help="path to input file 2")
parser.add_argument("--metric", type=str, help="metric for scoring (baseline, mask-random, mask-predict)")
parser.add_argument("--lm_model", type=str, help="pretrained LM model to use")

args = parser.parse_args()
evaluate(args)
