import tensorflow as tf
import numpy as np


x_data = []
for i in [8, 8, 4, 6, 11, 4, 4, 6, 8, 8]:
    batch_x = np.random.normal(size=[i, 8, 11])
    x_data.append(batch_x)

y_data = [7, 2, 5, 4, 5, 8, 7, 11, 3, 5]
x_data = np.array(x_data)
y_data = np.array(y_data)

X = tf.placeholder(tf.float32, [None, None, 8, 11])  # (?, ?, 8, 11)
Y = tf.placeholder(tf.float32, [None, ])             # (?, )
                                                                       #  X (?, ?, 8, 11)
W_conv1 = tf.Variable(tf.random_normal([2, 3, 11, 32], stddev=0.01))      # (2, 3, 11, 32)
h_conv1 = tf.nn.conv2d(X, W_conv1, strides=[1, 1, 1, 1], padding='SAME')  # (?, ?, 8, 32)

W_conv2 = tf.Variable(tf.random_normal([2, 2, 32, 64], stddev=0.01))            # (2, 2, 32, 64)
h_conv2 = tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')  # (?, ?, 8, 64)

w = tf.Variable(tf.constant(1., shape=[1, 1, 11, 64]))
DeConnv1 = tf.nn.conv2d_transpose(h_conv2, filter=w, output_shape=tf.shape(X), strides=[1, 2, 2, 1], padding='SAME')

final = DeConnv1   # (?, ?, 8, 11)
final_w = tf.Variable(tf.random_normal([2, 3, 11, 1], stddev=0.01))

final_conv = tf.nn.conv2d(final, final_w, strides=[1, 1, 1, 1], padding='SAME')
y = tf.reshape(final_conv, [-1, ])
print("y shape", y.shape)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y))

batch_x = {}
batch_y = {}
for i, x in enumerate(x_data):
    if len(x) not in batch_x.keys():
        batch_x[len(x)] = [x]
        batch_y[len(x)] = [y_data[i]]
    else:
        batch_x[len(x)].append(x)
        batch_y[len(x)].append(y_data[i])


# -- Initialize
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for key in batch_x.keys():
    mini_x = np.array(batch_x[key])
    mini_y = np.array(batch_y[key])
    deconv_val, final_conv_val, loss = sess.run([DeConnv1, final_conv, cross_entropy],
                                                feed_dict={X: mini_x, Y: mini_y})
    print(deconv_val.shape)
    print(final_conv.shape)
    print(loss)

