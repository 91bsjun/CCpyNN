import tensorflow as tf
import numpy as np

shapes = [(4, 4, 8, 92), (3, 3, 7, 92), (4, 4, 8, 92), (3, 3, 7, 92)]
x_data = []
for s in shapes:
    x = np.zeros(shape=s)
    print(x.shape)
    x_data.append(x)

X = tf.placeholder(tf.float32, [None, None, None, None, 92])
with tf.Session() as sess:
    sess.run(X, feed_dict={X: x_data})