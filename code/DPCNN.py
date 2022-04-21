import tensorflow as tf
class DPCNN:
    def build_model(self,emb, seq_len):

        self.channel_size = 250
        self.blocks_num = 9

        input_layer = tf.keras.layers.Input(shape=(seq_len, ), name='input')
        emb_layer = tf.keras.layers.Embedding(emb.shape[0], emb.shape[1], weights=[emb],
                                              trainable=False, input_length=seq_len, name='emb_w2v')(input_layer)
        x = tf.keras.layers.Conv1D(filters=self.channel_size, kernel_size=3, padding='same')(emb_layer)
        x = tf.keras.layers.LayerNormalization()(x)


        for i in range(self.blocks_num):
            x = self.__block(x)

        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Flatten()(x)
        out = tf.keras.layers.Dense(14, activation='softmax')(x)

        model = tf.keras.models.Model(inputs=input_layer, outputs=out)
        model.summary()
        model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

        return model

    def __block(self, px):

        x = tf.keras.layers.Activation('relu')(px)
        x = tf.keras.layers.Conv1D(filters=self.channel_size, kernel_size=3, padding='same')(x)
        x = tf.keras.layers.LayerNormalization()(x)

        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv1D(filters=self.channel_size, kernel_size=3, padding='same')(x)
        x = tf.keras.layers.LayerNormalization()(x)

        x = tf.keras.layers.add([px, x])
        tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='same')(x)

        return x