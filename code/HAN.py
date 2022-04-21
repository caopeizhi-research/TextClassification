import os
import random
import logging
import numpy as np
import pandas as pd

from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing import text, sequence
from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from transformers import BasicTokenizer

seed = 2022
random.seed(seed)
np.random.seed(seed)

df_train = pd.read_csv('./data/train_set.csv', sep='\t')
df_test =  pd.read_csv('./data/test_a.csv', sep='\t')
df_data = df_train.append(df_test)
df_data = df_data.reset_index(drop=True)
print(df_data.shape, df_train.shape)

max_words_num = None  # 不限制词表大小
seq_len = 2000
embedding_dim = 200
col = 'text'

print("Generate sequences")
os.makedirs('./data/seqs', exist_ok=True)  # 创建文件夹，在目录存在的情况下触发异常
seq_path = './data/seqs/seqs_{}_{}.npy'.format(max_words_num, seq_len)
word_index_path = './data/seqs/word_index_{}_{}.npy'.format(max_words_num, seq_len)

# 将每个句子处理为长度为256的片段
def sentence_split(document, max_sent_len=256, max_segment=16):
    document_len = len(document)
    index = list(range(0, document_len, max_sent_len))
    index.append(document_len)

    segments = []
    for i in range(len(index) - 1):
        segment = document[index[i]: index[i + 1]]
        len_seg = len(segment)
        segment = sequence.pad_sequences(segment, maxlen=max_sent_len, padding='post')
        # 每个元素包含句子长度和句子成分
        segments.append([len_seg, segment])

    if len(segments) > max_segment:
        segment_ = max_segment // 2
        return segments[:segment_] + segments[-segment_:]
    else:
        return sequence.pad_sequences(segments, maxlen=max_segment, padding='post', value=[0, [0] * max_segment])


if not os.path.exists(seq_path) or not os.path.exists(word_index_path):
    tokenizer = text.Tokenizer(num_words=max_words_num, lower=False, filters='', split=' ', oov_token=None)
    tokenizer.fit_on_texts(df_data[col].values.tolist())
    ids_doc = tokenizer.texts_to_sequences(df_data[col].values.tolist())

    # pre_post = [doc if len(doc) <= 2000 else doc[:1000] + doc[-1000:] for doc in ids_doc]
    # seqs = sequence.pad_sequences(pre_post, maxlen=seq_len, padding='post', truncating='pre')
    seqs = [sentence_split(doc, 256, 8) for doc in ids_doc]
    word_index = tokenizer.word_index

    np.save(seq_path, seqs)
    np.save(word_index_path, word_index)
else:
    seqs = np.load(seq_path)
    word_index = np.load(word_index_path, allow_pickle=True).item()

embedding_path = './emb/word2vec.txt'