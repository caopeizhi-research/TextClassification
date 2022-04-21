import tensorflow as tf
class Query(tf.Module):
    def __init__(self, in_features):
        super(Query, self).__init__()
        self.query = tf.Variable(initial_value=tf.random.normal([in_features, 1], mean=0.0, stddev=0.05), trainable=True, name='query')
    def __call__(self):
        return self.query
class Query2(tf.Module):
    def __init__(self, in_features):
        super(Query2, self).__init__()
        self.query = tf.Variable(initial_value=tf.random.normal([in_features, 1], mean=0.0, stddev=0.05), trainable=True, name='query2')
    def __call__(self):
        return self.query
class lstm_han:
    def __init__(self, dropout=0.3, word_dims=100, word_hidden_size=128, sent_hidden_size=256):
        self.dropout = dropout
        self.word_dims = word_dims
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size

        # 矩阵转置操作
        # x = tf.transpose(x, perm=[0, 2, 1])
    def build_model(self, vocab, extword_embed):
        # batch_inputs[batch_inputs1, batch_inputs2]: b x doc_len x sent_len
        batch_inputs = tf.keras.layers.Input(shape=(8, 768), name='inputs')

        batch_words, batch_extwords, batch_masks = batch_inputs[:, :, :256], batch_inputs[:, :, 256:512], batch_inputs[:, :, 512:]

        # batch_words, batch_extwords, batch_masks = batch_inputs[0], batch_inputs[1],batch_inputs[2]
        batch_size, max_doc_len, max_sent_len = tf.shape(batch_words)[0], tf.shape(batch_words)[1], tf.shape(batch_words)[2]
        batch_words = tf.reshape(batch_words, [-1, max_sent_len])
        batch_extwords = tf.reshape(batch_extwords, [-1, max_sent_len])
        batch_masks = tf.reshape(batch_masks, [-1, max_sent_len])

        batch_hiddens = self.WordLSTMEncoder(vocab, extword_embed, batch_words, batch_extwords, batch_masks) #sen_num*sent_len*sent_rep_size
        sent_reps = self.attention(batch_hiddens, batch_masks, self.word_hidden_size * 2) #sen_num*sent_rep_size

        batch_masks = tf.reshape(batch_masks, [batch_size, max_doc_len, max_sent_len])  # b x doc_len x max_sent_len
        sent_reps = tf.reshape(sent_reps, [batch_size, max_doc_len, self.sent_hidden_size]) # b x doc_len x sent_rep_size
        sent_masks = tf.cast(tf.reduce_any(tf.cast(batch_masks, tf.bool), axis=2), tf.float32)  # b x doc_len

        # doc_rep_size = sent_hidden_size * 2
        sent_hiddens = self.SentEncoder(sent_reps, sent_masks)  # b x doc_len x doc_rep_size
        doc_reps = self.attention2(sent_hiddens, sent_masks, self.sent_hidden_size * 2) # b * doc_rep_size

        batch_outputs = tf.keras.layers.Dense(14, activation='softmax', name='dense')(doc_reps)

        model = tf.keras.models.Model(inputs=batch_inputs, outputs=batch_outputs)
        model.summary()
        model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(),
                      metrics=['accuracy'])
        return model


    def attention(self, batch_hidden, batch_masks, hidden_size):
        """
        :param batch_hidden: b * len * hidden_size
        :param batch_masks: b * len
        :param hidden_size: lstm_hidden_size * 2
        :return:
        """

        key = tf.keras.layers.Dense(hidden_size, name='key')(batch_hidden)
        query = Query(hidden_size)()
        # 计算注意力
        outputs = tf.matmul(key, query)  # b * len
        outputs = tf.squeeze(outputs, axis=-1)
        outputs += (1.0 - batch_masks) * -10000.0  # masked

        attn_scores = tf.nn.softmax(outputs, axis=1, name='sm_1')
        attn_scores *= batch_masks  # 对于全零向量-10000的结果为1/len

        # sum weighted sources
        batch_outputs = tf.matmul(tf.expand_dims(attn_scores, axis=1), batch_hidden)  # b * hiddens
        batch_outputs = tf.squeeze(batch_outputs, axis=1)

        return batch_outputs

    def attention2(self, batch_hidden, batch_masks, hidden_size):
        """
        :param batch_hidden: b * len * hidden_size
        :param batch_masks: b * len
        :param hidden_size: lstm_hidden_size * 2
        :return:
        """

        key = tf.keras.layers.Dense(hidden_size, name='key2')(batch_hidden)
        query = Query2(hidden_size)()
        # 计算注意力
        outputs = tf.matmul(key, query)  # b * len
        outputs = tf.squeeze(outputs, axis=-1)
        outputs += (1.0 - batch_masks) * -10000.0  # masked

        attn_scores = tf.nn.softmax(outputs, axis=1, name='sm_12')
        attn_scores *= batch_masks  # 对于全零向量-10000的结果为1/len

        # sum weighted sources
        batch_outputs = tf.matmul(tf.expand_dims(attn_scores, axis=1), batch_hidden)  # b * hiddens
        batch_outputs = tf.squeeze(batch_outputs, axis=1)

        return batch_outputs
    # build word encoder
    def WordLSTMEncoder(self, vocab, extword_embed, word_ids, extword_ids, batch_masks):
        """
        :param extword_embed:
        :param word_ids: sen_num x sent_len
        :param extword_ids: sen_num x sent_len
        :param batch_masks: sen_num x sent_len
        :return:
        """

        word_embed = tf.keras.layers.Embedding(vocab.word_size, self.word_dims, name='word_embed', mask_zero=True)(word_ids)
        extword_embed = tf.keras.layers.Embedding(extword_embed.shape[0], extword_embed.shape[1], weights=[extword_embed],
                                                  trainable=False, name='ext_embed', mask_zero=True)(extword_ids)
        batch_embed = tf.keras.layers.add([word_embed, extword_embed], name='add1')
        batch_embed = tf.keras.layers.Dropout(self.dropout, name='drop1')(batch_embed)

        # sen_num x sent_len x  hidden*2
        hiddens = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(self.word_hidden_size, return_sequences=True, name='gru_1'), name='bi_1')(batch_embed)
        hiddens = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(self.word_hidden_size, return_sequences=True, name='gru_2'), name='bi_2')(hiddens)
        hiddens *= tf.expand_dims(batch_masks, axis=2)

        hiddens = tf.keras.layers.Dropout(self.dropout, name='drop_2')(hiddens)

        return hiddens

    # build sentence encoder
    def SentEncoder(self, sent_reps, sent_masks):
        """
        :param sent_reps: b x doc_len x sent_rep_size
        :param sent_masks: b x doc_len
        """

        hiddens = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(self.sent_hidden_size, return_sequences=True, name='gru3'), name='bi_3')(sent_reps)
        hiddens = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(self.sent_hidden_size, return_sequences=True, name='gru4'), name='bi_4')(hiddens)
        hiddens *= tf.expand_dims(sent_masks, axis=2)

        hiddens = tf.keras.layers.Dropout(self.dropout)(hiddens)

        return hiddens
