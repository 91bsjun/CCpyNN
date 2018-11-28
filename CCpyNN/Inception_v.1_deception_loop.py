import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.slim as slim

from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import IStructure

from CCpyNN.StructureEncoder import structure_encoder

np.set_printoptions(threshold=np.nan)
pd.set_option('display.max_rows', None)


def get_data(normalize_y, sample_size):
    radius = 3
    max_neighbor_num = 10
    training = 0.9
    test = 0.1

    df = pd.read_csv("./Data/metal-alloy-db.v2/00Total_DB.csv")
    # df = df.sample(n=len(df))
    df = df.sample(n=sample_size)

    cifs = "./Data/metal-alloy-db.v2/" + df['DBname'] + ".cif"
    structures = [IStructure.from_file(cif) for cif in cifs]
    encoded_structures = [structure_encoder(structure, radius, max_neighbor_num) for structure in structures]
    x_data = np.array(encoded_structures)

    formation_energy = df['FormationEnergy']
    if normalize_y:
        mean = formation_energy.mean()
        std = formation_energy.std()
        norm_form_energy = (df['FormationEnergy'] - mean) / std

        def norm_back(val, mean, std):
            return val * std + mean

        y_data = [[val] for val in norm_form_energy]
    else:
        y_data = [[val] for val in formation_energy]
    y_data = np.array(y_data)

    total = len(df)
    train = int(float(total) * training)
    test = int(float(total) * test)
    x_train = x_data[:train]
    y_train = y_data[:train]
    x_test = x_data[train:train + test]
    y_test = y_data[train:train + test]

    return x_train, y_train, x_test, y_test

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
    input_layer = tf.layers.conv3d(inputs=input_layer, filters=32,
                                   kernel_size=[2, 2, 4],
                                   padding="valid",
                                   strides=[2, 2, 4])
    fea_len = input_layer.shape[3]
    while fea_len != 1:
        input_layer = tf.layers.conv3d(inputs=input_layer, filters=64,
                                       kernel_size=[2, 2, 4],
                                       padding="same",
                                       strides=[1, 1, 4])
        fea_len = input_layer.shape[3]


    final_shape = input_layer.shape

    post_conv = tf.reshape(input_layer, [-1, final_shape[1], final_shape[2], final_shape[4]])

    pooling_layer = tf.layers.max_pooling2d(inputs=post_conv, pool_size=[2, 2], strides=2)

    return pooling_layer


def stem(input_layer):
    # Layer A : 1x1x1
    layer_A = tf.layers.conv3d(inputs=input_layer, filters=32,
                               kernel_size=[1, 1, 1],
                               padding="same",
                               strides=[1, 1, 1],
                               activation=tf.nn.relu)

    return layer_A


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
    fig = plt.figure(figsize=(10, 4))
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
    X = tf.placeholder(tf.float32, [None, 8, 10, 135])  # (?, N, 10, 225)
    # X = tf.placeholder(tf.float32, [None, None, 10, 225])  # (?, N, 10, 225)
    Y = tf.placeholder(tf.float32, [None, 1])  # (?, 1)
    keep_prob = tf.placeholder(tf.float32)
    is_train = tf.placeholder(tf.bool)

    fcX = tf.layers.dense(X, units=64)
    bnX = tf.layers.batch_normalization(fcX)
    expX = tf.expand_dims(bnX, axis=4)

    # inception_layer_1 = inception1(expX)

    deception_layer_1 = deception(expX)

    # flat_layer = tf.reshape(deception_layer_1, [-1, 2 * 2 * 256])
    flat_layer = tf.layers.flatten(deception_layer_1)

    dense_layer = tf.layers.dense(inputs=flat_layer, units=1024)

    dropout_layer = tf.nn.dropout(dense_layer, keep_prob=keep_prob)

    logit_layer = tf.layers.dense(dropout_layer, units=1)

    cost = tf.losses.mean_squared_error(Y, logit_layer)
    cost = tf.reduce_mean(cost)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    # parameters
    sample_size = 9000
    epoch_size = 50
    batch_size = 200
    train_loss = []

    with tf.Session() as sess:
        # -- Early stop code
        early_stopping = EarlyStopping(patience=5, verbose=1)
        # -- Initialize
        init = tf.global_variables_initializer()
        sess.run(init)

        x_train, y_train, x_test, y_test = get_data(normalize_y=False, sample_size=sample_size)
        batch_x, batch_y = get_batch_data(x_train, y_train, batch_size)
        print("Learning Start")
        for epoch in range(epoch_size):
            avg_cost = 0
            for i in range(len(batch_x)):
                c, hy, _ = sess.run([cost, logit_layer, optimizer], feed_dict={X: batch_x[i], Y: batch_y[i], keep_prob: 0.8})
                avg_cost += c
            avg_cost = avg_cost / len(batch_x)
            train_loss.append(avg_cost)
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

            if early_stopping.validate(avg_cost):
                break
        # -- Eval
        c, hy, _ = sess.run([cost, logit_layer, optimizer], feed_dict={X: x_test, Y: y_test, keep_prob: 1})
        print("Evaluated MSE : %.9f" % c)
        prd = np.array(hy).squeeze()
        cal = np.array(y_test).squeeze()

        plt = plot_result(train_loss, prd, cal)
        plt.show()
