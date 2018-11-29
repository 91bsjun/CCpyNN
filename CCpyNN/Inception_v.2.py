import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.slim as slim

from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import IStructure

try:
    from CCpyNN.HierarchicalCrystal import StructureToMatrixEncoder
except:
    from HierarchicalCrystal import StructureToMatrixEncoder

np.set_printoptions(threshold=np.nan)
pd.set_option('display.max_rows', None)


def get_data(normalize_y, sample_size):
    training = 0.9
    test = 0.1

    df = pd.read_csv("./Data/metal-alloy-db.v1/00Total_DB.csv")
    # df = df.sample(n=len(df))
    df = df.sample(n=sample_size)

    total = len(df)
    train = int(float(total) * training)
    test = int(float(total) * test)

    train_df = df[:train]
    test_df = df[train:train + test]

    def get_group(df):
        formation_energy = np.array(df['FormationEnergy'].tolist())
        cifs = "./Data/metal-alloy-db.v1/" + df['DBname'] + ".cif"
        batch_group_x = {}
        batch_group_y = {}
        for i, cif in enumerate(cifs):
            encoder = StructureToMatrixEncoder(cif)
            m = encoder.get_structure_matrix()
            if m.shape not in batch_group_x.keys():
                batch_group_x[m.shape] = [m]
                batch_group_y[m.shape] = [[formation_energy[i]]]
            else:
                batch_group_x[m.shape].append(m)
                batch_group_y[m.shape].append([formation_energy[i]])

        return batch_group_x, batch_group_y

    train_x, train_y = get_group(train_df)
    test_x, test_y = get_group(test_df)

    return train_x, train_y, test_x, test_y


def get_batch_data(X, Y, batch_size):
    total_data = len(X)
    batch_x = []
    batch_y = []
    if total_data < batch_size:
        batch_x.append(X)
        batch_y.append(Y)
    else:
        batch_group = int(total_data / batch_size)
        for b in range(batch_group):
            batch_x.append(X[b * batch_size:(b + 1) * batch_size])
            batch_y.append(Y[b * batch_size:(b + 1) * batch_size])
        if (b+1) * batch_size != total_data:
            batch_x.append(X[(b + 1) * batch_size:])
            batch_y.append(Y[(b + 1) * batch_size:])

    return batch_x, batch_y


def inception1(input_layer):
    # Layer A : 1x1x1
    layer_A = tf.layers.conv3d(inputs=input_layer, filters=32,
                               kernel_size=[1, 1, 1],
                               padding="same",
                               strides=[1, 1, 1],
                               activation=tf.nn.relu)

    # Layer B : 1x1x1 -> 2x2x4
    layer_B = tf.layers.conv3d(inputs=input_layer, filters=16,
                               kernel_size=[1, 1, 1],
                               padding="same",
                               strides=[1, 1, 1],
                               activation=tf.nn.relu)
    layer_B = tf.layers.conv3d(inputs=layer_B, filters=32,
                               kernel_size=[2, 2, 4],
                               padding="same",
                               strides=[1, 1, 1],
                               activation=tf.nn.relu)

    # Layer C : 1x1x1 -> 4x4x8
    layer_C = tf.layers.conv3d(inputs=input_layer, filters=16,
                               kernel_size=[1, 1, 1],
                               padding="same",
                               strides=[1, 1, 1],
                               activation=tf.nn.relu)
    layer_C = tf.layers.conv3d(inputs=layer_C, filters=32,
                               kernel_size=[4, 4, 8],
                               padding="same",
                               strides=[1, 1, 1],
                               activation=tf.nn.relu)

    # Layer D : Pooling -> 1x1x1
    layer_D = tf.layers.max_pooling3d(inputs=input_layer,
                                      pool_size=[1, 3, 1],
                                      strides=[1, 1, 1],
                                      padding='same')
    layer_D = tf.layers.conv3d(inputs=layer_D, filters=32,
                               kernel_size=[1, 1, 1],
                               padding="same",
                               strides=[1, 1, 1],
                               activation=tf.nn.relu)

    # Concat Layer
    concat_layer = tf.concat([layer_A, layer_B, layer_C, layer_D], axis=4)

    return concat_layer


def deception(input_layer):
    layer_1 = tf.layers.conv2d(inputs=input_layer, filters=32,
                               kernel_size=[2, 2],
                               padding="valid",
                               strides=[2, 2])

    layer_2 = tf.layers.conv2d(inputs=layer_1, filters=64,
                               kernel_size=[5, 5],
                               padding="same",
                               strides=[1, 1])

    layer_3 = tf.layers.conv2d(inputs=layer_2, filters=128,
                               kernel_size=[3, 3],
                               padding="valid",
                               strides=[1, 1])

    layer_4 = tf.layers.conv2d(inputs=layer_3, filters=256,
                               kernel_size=[2, 2],
                               padding="valid",
                               strides=[2, 2])

    # layer_5 = tf.layers.conv2d(inputs=layer_4, filters=128,
    #                            kernel_size=[2, 2],
    #                            padding="valid",
    #                            strides=[2, 2])

    # post_conv = tf.reduce_mean(layer_5, axis=1)
    # s = post_conv.shape

    pooling_layer = tf.layers.max_pooling2d(inputs=layer_4, pool_size=[2, 2], strides=2)

    return pooling_layer



def stem(input_layer):
    '''
    input_layer = (?, ?, 92)
    fcX = tf.layers.dense(input_layer, units=128)
    bnX = tf.layers.batch_normalization(fcX)
    expX = tf.expand_dims(bnX, axis=4)
    '''
    bnX = tf.layers.batch_normalization(input_layer)
    fcX = tf.layers.dense(bnX, units=64)
    expX = tf.expand_dims(fcX, axis=3)
    # layer_1 = tf.layers.conv2d(inputs=expX, filters=32,
    #                            kernel_size=[1, 1],
    #                            padding="same",
    #                            strides=[1, 1])

    return expX


class EarlyStopping():
    def __init__(self, patience=0, verbose=0):
        self._step = 0
        self._loss = float('inf')
        self.patience = patience
        self.verbose = verbose

    def validate(self, loss):
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print(f'Training process is stopped early....')
                return True
        else:
            self._step = 0
            self._loss = loss

        return False


def plot_result(loss, prd, cal):
    min_val = min(prd.min(), cal.min())
    max_val = max(prd.max(), cal.max())

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    # plot train loss
    plt.subplot(1, 2, 1)
    plt.plot([x + 1 for x in range(len(loss))], loss)
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Loss (MSE)", fontsize=16)

    # plot prd vs calc
    plt.subplot(1, 2, 2)
    plt.scatter(prd, cal, s=10)
    plt.plot([min_val, max_val], [min_val, max_val], color='r')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.xlabel("Predicted", fontsize=16)
    plt.ylabel("Calculated", fontsize=16)

    return plt


if __name__ == "__main__":
    # parameters
    sample_size = 4000
    epoch_size = 50
    batch_size = 150
    train_loss = []

    X = tf.placeholder(tf.float32, [None, 4, 4, 8, 92])
    rsX = tf.reshape(X, [-1, 4 * 4 * 8, 92])
    Y = tf.placeholder(tf.float32, [None, 1])  # (?, 1)
    keep_prob = tf.placeholder(tf.float32)

    # ------- Stem -------- #
    stem_layer = stem(rsX)

    # ----- Deception ----- #
    deception_layer_1 = deception(stem_layer)

    #flat_layer = tf.reshape(deception_layer_1, [-1, 2 * 2 * 256])
    flat_layer = tf.layers.flatten(deception_layer_1)

    dense_layer = tf.layers.dense(inputs=flat_layer, units=1024)

    dropout_layer = tf.nn.dropout(dense_layer, keep_prob=keep_prob)

    logit_layer = tf.layers.dense(dropout_layer, units=1)

    cost = tf.losses.mean_squared_error(Y, logit_layer)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)



    with tf.Session() as sess:
        # -- Early stop code
        # early_stopping = EarlyStopping(patience=5, verbose=1)
        # -- Initialize
        init = tf.global_variables_initializer()
        sess.run(init)

        x_train, y_train, x_test, y_test = get_data(normalize_y=False, sample_size=sample_size)
        print("Learning Start")
        for epoch in range(epoch_size):
            avg_cost = 0
            total_len = 0
            # for key in x_train.keys():
            mini_x = np.array(x_train[(4, 4, 8, 92)], dtype='float32')
            mini_y = np.array(y_train[(4, 4, 8, 92)], dtype='float32')

            c, _ = sess.run([cost, optimizer], feed_dict={X: mini_x, Y: mini_y, keep_prob: 0.8})

            train_loss.append(c)
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(c))
            # if early_stopping.validate(avg_cost):
            #     break

        # -- Eval
        mini_test_x = np.array(x_test[(4, 4, 8, 92)], dtype='float32')
        mini_test_y = np.array(y_test[(4, 4, 8, 92)], dtype='float32')
        c, hy, _ = sess.run([cost, logit_layer, optimizer], feed_dict={X: mini_test_x, Y: mini_test_y, keep_prob: 1})
        print("Evaluated MSE : %.9f" % c)
        prd = np.array(hy).squeeze()
        cal = np.array(mini_test_y).squeeze()
        print(prd)
        print(cal)

        plt = plot_result(train_loss, prd, cal)
        plt.show()
