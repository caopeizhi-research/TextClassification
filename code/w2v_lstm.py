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

# 并没有设置未知单词，所有单词都见过，0是padding值
# 前后截取2000个字符，虽然截取3000会更好一点，但是训练时间会大大增加
if not os.path.exists(seq_path) or not os.path.exists(word_index_path):
    # num_words: 需要保留的最大词数，基于词频。只有最常出现的num_words词会被保留
    # filters: 一个字符串，其中每个元素是一个将从文本中过滤掉的字符。默认值是所有标点符号，加上制表符和换行符
    # 0是不会被分配给任何单词的保留索引
    # split: 按该字符串切割文本
    # oov_token: 如果给出，它将被添加到 word_index 中，并用于在 text_to_sequence 调用期间替换词汇表外的单词
    tokenizer = text.Tokenizer(num_words=max_words_num, lower=False, filters='', split=' ', oov_token=None)
    tokenizer.fit_on_texts(df_data[col].values.tolist())
    # texts_to_sequences(texts)返回值：序列的列表，列表中每个序列对应于一段输入文本, 将此词转换为了id
    ids_doc = tokenizer.texts_to_sequences(df_data[col].values.tolist())
    # 截取前后共2000单词
    pre_post = [doc if len(doc) <= 2000 else doc[:1000] + doc[-1000:] for doc in ids_doc]
    # sequence.pad_sequences将多个序列截断或补齐为相同长度
    # padding: 字符串，‘pre’ 或 ‘post’ ，在序列的前端补齐还是在后端补齐
    # truncating: 字符串，‘pre’ 或 ‘post’ ，移除长度大于 maxlen 的序列的值，要么在序列前端截断，要么在后端
    # value: 浮点数，表示用来补齐的值，默认应该为0
    seqs = sequence.pad_sequences(pre_post, maxlen=seq_len, padding='post', truncating='pre')
    word_index = tokenizer.word_index

    np.save(seq_path, seqs)
    np.save(word_index_path, word_index)
else:
    seqs = np.load(seq_path)
    word_index = np.load(word_index_path, allow_pickle=True).item()

embedding_path = './emb/word2vec.txt'
# 加载和操作词向量模型的工具
model = KeyedVectors.load_word2vec_format(embedding_path)

embedding = np.zeros((len(word_index) + 1, embedding_dim))
# tqdm 可以在Python长循环中添加一个进度提示信息，用户只需要封装任意的迭代器tqdm(iterator)
for word, i in tqdm(word_index.items()):
    embedding_vector = model[word] if word in model else None
    if embedding_vector is not None:
        embedding[i] = embedding_vector
embedding = embedding / np.std(embedding)


all_index = df_data[df_data['label'].notnull()].index.tolist()
test_index = df_data[df_data['label'].isnull()].index.tolist()

# 因为后面要load_weights，所以为每一层命名了
def build_model(emb, seq_len):
    # ‘name’, 层的名字，自定义层时不需自定义初始化name,只需调Layer这个父类的初始化方法即可，它会为Layer实例绑定name属性
    # ‘trainable’, 是否可训练。这个值将覆盖Layer类中add_weight方法的trainable参数的值
    # ‘weights’, 为该层权重指定任意自定义的初始值。如果embeddings_initializer被给定了，那么weights将覆盖它。weights的自定义性更广。
    # 这里虽然设成了False，但是根据你词向量的具体情况，可能设成True会更好
    emb_layer = tf.keras.layers.Embedding(emb.shape[0], emb.shape[1], weights=[emb],
                                          trainable=False, input_length=seq_len, name='emb_word2vec')

    seq = tf.keras.layers.Input(shape=(seq_len, ), name='seq_input')
    seq_emb = emb_layer(seq)
    # 会将某个特定的维度全部置0SpatialDropout1D与Dropout的作用类似，但它断开的是整个1D特征图，而不是单个神经元。如果一张特征图的相邻像素之间有
    # 很强的相关性（通常发生在低层的卷积层中），那么普通的dropout无法正则化其输出，否则就会导致明显的学习率下降。这种情况下，SpatialDropout1D能
    # 够帮助提高特征图之间的独立性，应该用其取代普通的Dropout
    seq_emb = tf.keras.layers.SpatialDropout1D(rate=0.2, name='drop_out1')(seq_emb)

    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(200, return_sequences=True, name='lstm'), name='bi_layer')(seq_emb)
    lstm_avg_pool = tf.keras.layers.GlobalAveragePooling1D(name='avg')(lstm)
    lstm_max_pool = tf.keras.layers.GlobalMaxPool1D(name='max')(lstm)

    x = tf.keras.layers.concatenate([lstm_avg_pool, lstm_max_pool], name='concat')
    # 为什么没加激活层 LSTM里已经激活过了
    x = tf.keras.layers.Dense(1024, name='dense_1')(x)
    x = tf.keras.layers.BatchNormalization(name='bn')(x)
    x = tf.keras.layers.Activation(activation='relu', name='acti')(x)
    x = tf.keras.layers.Dropout(0.2, name='drop_2')(x)
    out = tf.keras.layers.Dense(14, activation='softmax', name='dense_2')(x)

    model = tf.keras.models.Model(inputs=seq, outputs=out)
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

    return model

# 定义一个回调类
class Evaluator(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.best_val_f1 = 0.
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def evaluate(self):
        y_true = self.y_val
        y_pred = self.model.predict(self.x_val).argmax(axis=1)
        f1 = f1_score(y_true, y_pred, average='macro')
        return f1

    def on_epoch_end(self, epoch, logs=None):
        val_f1 = self.evaluate()
        if val_f1 > self.best_val_f1:
            self.best_val_f1 = val_f1
        logs['val_f1'] = val_f1
        print(f'val_f1:{val_f1:.5f}, best_val_f1:{self.best_val_f1:.5f}')


bs = 256
epochs = 30
monitor = 'val_f1'


Kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
for fold_id, (train_index, val_index) in enumerate(Kfold.split(all_index, df_data.iloc[all_index]['label'])):
    train_x = seqs[train_index]
    val_x = seqs[val_index]

    label = df_data['label'].values
    train_y = label[train_index]
    val_y = label[val_index]

    model_path = './model/lstm_{}.h5'.format(fold_id)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path, monitor=monitor, verbose=1, save_best_only=True, mode='max', save_weights_only='True')
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=5, verbose=1, mode='max')
    # 当发现和上个epoch相比没有上升，则经过patience个epoch停止
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=2, mode='max', verbose=1)

    model = build_model(embedding, seq_len)
    # 细思极恐，相当于预训练过了吗
    # model.load_weights(model_path, by_name=True)
    model.fit(train_x, train_y, batch_size=bs, epochs=epochs,
              validation_data=(val_x, val_y), verbose=1, shuffle=True,
              callbacks=[Evaluator(validation_data=(val_x, val_y)), checkpoint, reduce_lr, earlystopping])


# 模型预测
# oof_pred = np.zeros((len(all_index), 14))
test_pred = np.zeros((len(test_index), 14))

for fold_id, (train_index, val_index) in enumerate(Kfold.split(all_index, df_data.iloc[all_index]['label'])):
    model = build_model(embedding, seq_len)
    model_path = './model/lstm_{}.h5'.format(fold_id)
    model.load_weights(model_path, by_name=True)

    # val_x = seqs[val_index]
    # prob = model.predict(val_x, batch_size=bs, verbose=1)
    # oof_pred[val_index] = prob

    test_x = seqs[test_index]
    prob = model.predict(test_x, batch_size=bs, verbose=1)
    df = pd.DataFrame(prob)
    df.to_csv('./sub/lstm_{}.csv'.format(fold_id), index=False, header=False, sep=',')
    test_pred += prob / 5

# df_oof = df_data.iloc[all_index][['label']]
# df_oof['predict'] = np.argmax(oof_pred, axis=1)
# f1score = f1_score(df_oof['label'], df_oof['predict'], average='macro')
# print(f1score)

# np.save('./prob/sub_5fold_lstm_{}.npy'.format(f1score), test_pred)
# np.save('./prob/oof_5fold_lstm_{}.npy'.format(f1score), oof_pred)

sub = pd.DataFrame()
sub['label'] = np.argmax(test_pred, axis=1)
sub.to_csv('./submit/lstm_{}.csv'.format(111), index=False)






# train_data = pd.read_csv('./data/train_set.csv', sep='\t')
# test_data = pd.read_csv('./data/test_a.csv', sep='\t')
#
# # logging.info('Start word to vector...')
# # train_w2v = list(map(lambda x: list(x.split()), train_texts))
# # # min_count (int, optional) – 忽略词频小于此值的单词
# # # sg ({0, 1}, optional) – 模型的训练算法: 1: skip-gram; 0: CBOW
# # # hs ({0, 1}, optional) – 1: 采用hierarchical softmax训练模型; 0: 使用负采样
# # # negative (int, optional) – > 0: 使用负采样，设置多个负采样(通常在5-20之间)
# # # sorted_vocab ({0, 1}, optional) – 如果为1，则在分配单词索引前按降序对词汇表进行排序
# # # window,一个句子中当前单词和被预测单词的最大距离。skipgram一般10左右，CBOW一般5左右
# # model = Word2Vec(train_w2v, vector_size=200, sg=0, hs=0, negative=5, sorted_vocab=1, min_count=5, sample=0.001, window=5)
# # model.save("./emb/model/word2vec.bin")
# # model = Word2Vec.load("./emb/model/word2vec.bin")   
# # model.wv.save_word2vec_format("./emb/model/word2vec.txt", binary=False)