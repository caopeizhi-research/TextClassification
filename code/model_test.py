import tensorflow as tf
p =[[[2, 2, 3],[1, 0, 0]]]
q = [[1],[1],[0]]
import numpy as np
a = np.array([[2, 2, 3],[1, 0, 0]])
b = np.array([[2, 2, 3],[1, 0, 0]])
c = tf.reshape(a, [-1])

d = tf.convert_to_tensor([[[1,1,0],[0,0,0]]])


class Query(tf.Module):
    def __init__(self, in_features):
        super(Query, self).__init__()
        self.query = tf.Variable(initial_value=tf.random.normal([in_features, 1], mean=0.0, stddev=0.05), trainable=True, name='query')
    def __call__(self):
        return self.query

query = Query(3)
print(query().trainable)
