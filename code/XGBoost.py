import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.utils import compute_sample_weight
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import plot_tree
from xgboost import plot_importance

time1 = time.time()

# 分隔符\t表示Tab建
train_df = pd.read_csv('./data/train_set.csv', sep='\t')
submit_df = pd.read_csv('./data/test_a.csv', sep='\t')
all_text = pd.concat([train_df['text'], submit_df['text']])
# # 分析句子长度
# train_df['text_len'] = train_df['text'].apply(lambda x: len(x.split()))
# print(train_df['text_len'].describe())
tfidf = TfidfVectorizer(ngram_range=(1, 3),
                        max_features=5000,
                        max_df=0.95)
tfidf.fit(all_text)
train_text = tfidf.transform(train_df['text'])
train_label = train_df['label'].tolist()
print(train_text.shape)

test_text = tfidf.fit_transform(submit_df['text'])
print("测试集形状：{}".format(test_text.shape))

time2 = time.time()
print("提取特征时间:{}分{}秒".format((time2 - time1) // 60, (time2 - time1) % 60))

x_train = train_text
y_train = train_label

# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=2022)

# 计算样本权重
sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)

# params = {
#     'booster': 'gbtree',
#     'n_estimator': 10000,
#     'objective': 'multi:softmax',   # 回归任务设置为：'objective': 'reg:gamma',
#     'num_class': 14,      # 回归任务没有这个参数
#     'gamma': 0.1,  # 最小分裂损失[0, 0.05 ~ 0.1, 0.3, 0.5, 0.7, 0.9, 1]
#     'max_depth': 3,  # 树的最大深度 越多越容易过拟合 [3, 5, 6, 7, 9, 12, 15, 17, 25]
#     'lambda': 2,  # L2正则化的权重[0, 0.1, 0.5, 1]
#     'subsample': 0.7,  # [0.6, 0.7, 0.8, 0.9, 1]
#     'colsample_bytree': 0.7,  # 每棵树选取的特征比例[0.6, 0.7, 0.8, 0.9, 1]
#     'min_child_weight': 3,  # 越大越保守，越不容易过拟合 [1, 3, 5, 7]
#     'verbosity': 1,
#     'eta': 0.1,  # 学习率，减少这个的同时需要增加基学习器的数量0.01\.015\0.025\0.05\0.1
#     'seed': 1000,
#     'nthread': 4,
#     'scale_posweight': 1
# }
# plst = list(params.items())

model = xgb.XGBClassifier(
    max_depth=10,
    learning_rate=0.1,  # 0.01\.015\0.025\0.05\0.1
    n_estimators=300,  # 使用多少个弱分类器
    objective='multi:softmax',
    num_class=14,
    booster='gbtree',
    gamma=0.1,
    min_child_weight=3,
    max_delta_step=0,
    subsample=0.7,   # [0.6, 0.7, 0.8, 0.9, 1]
    colsample_bytree=0.7,  # [0.6, 0.7, 0.8, 0.9, 1]
    reg_alpha=0,
    reg_lambda=2,
    seed=2022   # 随机数种子
)

# model.fit(x_train, y_train, sample_weight=sample_weight,
#           eval_set=[(x_train, y_train), (x_test, y_test)], eval_metric=['mlogloss'],
#           early_stopping_rounds=10, verbose=10)

model.fit(x_train, y_train, sample_weight=sample_weight,
          eval_set=[(x_train, y_train)], eval_metric=['mlogloss'],
          early_stopping_rounds=10, verbose=10)

time3 = time.time()

print("训练时间:{}分{}秒".format((time3 - time2) // 60, (time3 - time2) % 60))

fake_label = pd.read_csv('./submit/lstm_111.csv')
fake_label = fake_label['label']

# val_predict = model.predict(x_val)
# print(f1_score(y_val, val_predict, average='macro'))

# model.save_model('./model/xgboost.json')

submission = pd.read_csv('./data/test_a_sample_submit.csv')
preds = model.predict(test_text)
print(f1_score(fake_label, preds, average='macro'))
submission['label'] = preds
submission.to_csv('./submit/xgboost_submit.csv', index=False)

# plot_importance(model)
# plt.show()
