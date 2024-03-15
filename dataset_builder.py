import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pickle as pkl
import re

import pretty_midi
from gensim.models import KeyedVectors
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation, strip_numeric
import gensim.downloader as api

from sklearn.model_selection import train_test_split
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

import nltk
# nltk.download('punkt')
from nltk.tokenize import sent_tokenize

train_path = r'/home/munz/school/deep_learning/hw3/lyrics_train_set2.csv'
test_path = r'/home/munz/school/deep_learning/hw3/lyrics_test_set.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

train['artist'] = train['artist'].str.strip()
train['song'] = train['song'].str.strip()
test['artist'] = test['artist'].str.strip()
test['song'] = test['song'].str.strip()

word2vec_path = r'/home/munz/school/deep_learning/hw3/GoogleNews-vectors-negative300.bin.gz'
word2vec_model = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

bos_token = 'BOS '
eos_token = ' EOS '
eof_token = ' EOF'

def tokenize_lyrics(lyrics):
    lyrics = lyrics.lower()
    lyrics = lyrics.replace('&', eos_token)
    lyrics = bos_token + lyrics + eof_token
    lyrics = strip_punctuation(lyrics)
    lyrics = strip_numeric(lyrics)
    lyrics = re.sub(r'\(.*?\)', '', lyrics)
    lyrics = re.sub(r'\[.*?\]', '', lyrics)
    lyrics_len = len(lyrics.split())

    # tokenizing
    lyrics = lyrics.split()
    word_ids = [word2vec_model.key_to_index[word] for word in lyrics if word in word2vec_model]
    vectors = [word2vec_model[word] for word in lyrics if word in word2vec_model]

    oov_percentage = (lyrics_len - len(word_ids)) / lyrics_len
    return word_ids, vectors, oov_percentage

lyrics_dict = {}
for i, row in tqdm(train.iterrows(), total=len(train)):
    song_name = row['song']
    artist = row['artist']
    word_ids, vectors, oov_percentage = tokenize_lyrics(row['lyrics'])
    key = song_name + ' ' + artist
    lyrics_dict[key] = word_ids