
import pandas as pd
from tqdm import tqdm
import pickle as pkl
import re
from gensim.models import KeyedVectors
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation, strip_numeric
import gensim.downloader as api
import sys
sys.path.append('../')


train_path = r'data/lyrics_train_set2.csv'
test_path = r'data/lyrics_test_set.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

train['artist'] = train['artist'].str.strip()
train['song'] = train['song'].str.strip()
test['artist'] = test['artist'].str.strip()
test['song'] = test['song'].str.strip()

try:
    word2vec_path = r'models/GoogleNews-vectors-negative300.bin.gz'
    word2vec_model = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
except:
    try:
        word2vec_path = r'models/word2vec-google-news-300.model'
        word2vec_model = KeyedVectors.load(word2vec_path)
    except:
        word2vec_model = api.load('word2vec-google-news-300')
        word2vec_model.save(r'models/word2vec-google-news-300.model')

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

# save as pkl file
lyrics_pkl_path = r'data/lyrics_dict.pkl'
with open(lyrics_pkl_path, 'wb') as f:
    pkl.dump(lyrics_dict, f)