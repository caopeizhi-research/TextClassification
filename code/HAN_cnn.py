import os
import time
import logging

import random
import numpy as np
import pandas as pd
from Vocab import Vocab
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from HAN_model import lstm_han

seed = 2022
random.seed(seed)
np.random.seed(seed)

df_train = pd.read_csv('./data/train_set.csv', sep='\t', nrows=20000)
df_test = pd.read_csv('./data/test_a.csv', sep='\t', nrows=10000)

train_data = {'label': df_train['label'].tolist(), 'text': df_train['text'].tolist()}
test_data = {'label': [0] * len(df_test), 'text': df_test['text'].tolist()}

# build vocabulary
vocab = Vocab(train_data)

embeddings = vocab.load_pretrained_embs('./emb/word2vec/word2vec.txt')


# 最大文档长度、最大句子长度
max_segment = 8
max_sent_len = 256

# build dataset
def sentence_split(text, vocab, max_sent_len=256, max_segment=16):
    words = text.strip().split()
    document_len = len(words)

    index = list(range(0, document_len, max_sent_len))
    index.append(document_len)

    segments = []
    for i in range(len(index) - 1):
        segment = words[index[i]: index[i + 1]]
        assert len(segment) > 0
        segment = [word if word in vocab._id2word else '<UNK>' for word in segment]
        segments.append([len(segment), segment])

    assert len(segments) > 0
    if len(segments) > max_segment:
        segment_ = int(max_segment / 2)
        return segments[:segment_] + segments[-segment_:]
    else:
        return segments

def get_examples(data, vocab, max_sent_len=256, max_segment=8):
    label2id = vocab.label2id
    examples = []

    for text, label in zip(data['text'], data['label']):
        # label
        id = label2id(label)

        # words
        sents_words = sentence_split(text, vocab, max_sent_len, max_segment)
        doc = []
        for sent_len, sent_words in sents_words:
            word_ids = vocab.word2id(sent_words)
            extword_ids = vocab.extword2id(sent_words)
            doc.append([sent_len, word_ids, extword_ids])
        examples.append([id, len(doc), doc])

    logging.info('Total %d docs.' % len(examples))
    return examples

# 转换为模型可接收数据
def batch2tensor(batch_data):
    '''
        [[label, doc_len, [[sent_len, [sent_id0, ...], [sent_id1, ...]], ...]]
    '''
    batch_size = len(batch_data)
    doc_labels = []
    doc_lens = []
    doc_max_sent_len = []
    for doc_data in batch_data:
        doc_labels.append(doc_data[0])
        doc_lens.append(doc_data[1])
        sent_lens = [sent_data[0] for sent_data in doc_data[2]]
        max_sent_len = max(sent_lens)
        doc_max_sent_len.append(max_sent_len)

    max_doc_len = max(doc_lens)
    max_sent_len = max(doc_max_sent_len)

    batch_inputs1 = np.zeros((batch_size, max_doc_len, max_sent_len), dtype='float16')
    batch_inputs2 = np.zeros((batch_size, max_doc_len, max_sent_len), dtype='float16')
    batch_masks = np.zeros((batch_size, max_doc_len, max_sent_len), dtype='float16')

    for b in range(batch_size):
        for sent_idx in range(doc_lens[b]):
            sent_data = batch_data[b][2][sent_idx]
            for word_idx in range(sent_data[0]):
                batch_inputs1[b, sent_idx, word_idx] = sent_data[1][word_idx]
                batch_inputs2[b, sent_idx, word_idx] = sent_data[2][word_idx]
                batch_masks[b, sent_idx, word_idx] = 1

    return (batch_inputs1, batch_inputs2, batch_masks), doc_labels

print("Generate inputs")
os.makedirs('./data/HAN_inputs', exist_ok=True)
inputs_train = './data/HAN_inputs/te_inputs_train.npy'
inputs_test = './data/HAN_inputs/te_inputs_test.npy'





if not os.path.exists(inputs_train) or not os.path.exists(inputs_test):
    x_train, y_train = batch2tensor(get_examples(train_data, vocab, max_sent_len, max_segment))
    x_test, y_test = batch2tensor(get_examples(test_data, vocab, max_sent_len, max_segment))
    np.save(inputs_train, x_train)
    np.save(inputs_test, x_test)
else:
    x_train = np.load(inputs_train)
    y_train = np.array(train_data['label'])
    x_test = np.load(inputs_test)
    y_test = np.array(test_data['label'])

x_train = np.concatenate([x_train[0], x_train[1], x_train[2]], axis=-1)
print(x_train.shape)
print(y_train.shape)

bs = 16
epochs = 1
monitor = 'val_f1'

lstm_han = lstm_han()
model = lstm_han.build_model(vocab=vocab, extword_embed=embeddings)
model.fit(x_train, y_train, batch_size=bs, epochs=epochs, verbose=1, shuffle=True)