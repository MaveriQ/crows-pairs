from tokenizers import BertWordPieceTokenizer
from spacy.tokens import Doc
import spacy
import os
import requests

class BertTokenizer:
    def __init__(self, vocab, vocab_file, lowercase=True):
        self.vocab = vocab
        self._tokenizer = BertWordPieceTokenizer(vocab_file, lowercase=lowercase)

    def __call__(self, text):
        tokens = self._tokenizer.encode(text)
        words = []
        spaces = []
        for i, (text, (start, end)) in enumerate(zip(tokens.tokens, tokens.offsets)):
            words.append(text)
            if i < len(tokens.tokens) - 1:
                # If next start != current end we assume a space in between
                next_start, next_end = tokens.offsets[i + 1]
                spaces.append(next_start > end)
            else:
                spaces.append(True)
        return Doc(self.vocab, words=words, spaces=spaces)

nlp = spacy.load('en_core_web_lg')
url = "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt"

if not os.path.exists("bert-base-uncased-vocab.txt"):
    print(f"Downloading bert-base-uncased-vocab.txt")
    r = requests.get(url)
    with open('bert-base-uncased-vocab.txt', 'wb') as f:
        f.write(r.content)
    
nlp.tokenizer = BertTokenizer(nlp.vocab, "bert-base-uncased-vocab.txt")
