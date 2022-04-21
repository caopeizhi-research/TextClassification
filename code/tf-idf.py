from sklearn.utils import class_weight
import numpy as np
import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


train_set = pd.read_csv('./data/train_set.csv', sep='\t', encoding='UTF-8', nrows=20000)
test_set = pd.read_csv('./data/test_a.csv', sep='\t', encoding='UTF-8', nrows=20000)

x_train = train_set['text']
y_train = train_set['label']

x_test = test_set['text']
all_text = pd.concat([x_train, x_test])

print(x_train.shape, y_train.shape, x_test.shape, all_text.shape)

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=5000,
    max_df=0.95,
    analyzer='word',
    token_pattern=r'\w{1,}'
)

time1 = time.time()

tfidf.fit(all_text)

time2 = time.time()
print("提取特征时间:{}分{}秒".format((time2 - time1) // 60, (time2 - time1) % 60))

x_train = tfidf.transform(x_train)
x_test = tfidf.transform(x_test)
print(x_train.shape)

time3 = time.time()
print("提取特征时间:{}分{}秒".format((time3 - time1) // 60, (time3 - time1) % 60))