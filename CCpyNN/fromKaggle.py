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
            batch_x.append(X[b*batch_size:(b+1)*batch_size])
            batch_y.append(Y[b*batch_size:(b+1)*batch_size])
        if (b+1) * batch_size != total_data:
            batch_x.append(X[(b+1)*batch_size:])
            batch_y.append(Y[(b+1)*batch_size:])

    return batch_x, batch_y

if __name__ == "__main__":
    radius = 3
    max_neighbor_num = 10
    training = 0.9
    test = 0.1

    df = pd.read_csv("./Data/metal-alloy-db.v1/00Total_DB.csv")
    #df = df.sample(n=len(df))
    df = df.sample(n=1000)

    cifs = "./Data/metal-alloy-db.v1/" + df['DBname'] + ".cif"
    structures = [IStructure.from_file(cif) for cif in cifs]

    print("Encoding structures")
    import pickle
    if "encoded_structure.pkl" in os.listdir("./"):
        print("using pickle")
        with open("encoded_structure.pkl", "rb") as mydata:
            encoded_structures = pickle.load(mydata)
    else:
        encoded_structures = [structure_encoder(structure, radius, max_neighbor_num) for structure in structures]
        with open("encoded_structure.pkl", "wb") as savedata:
            pickle.dump(encoded_structures, savedata)

    print("Done")

    #encoded_structures = [structure_encoder(structure, radius, max_neighbor_num) for structure in structures]
    x_data = np.array(encoded_structures)
    x_data = np.expand_dims(x_data, axis=4)

    formation_energy = df['FormationEnergy']
    y_normalization = False
    if y_normalization:
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
    x_test = x_data[train:train+test]
    y_test = y_data[train:train+test]

    X = tf.placeholder(tf.float32, [None, 8, 10, 225, 1])  # (?, N, 10, 104)
    X1 = slim.fully_connected(X, 128)
    X2 = tf.contrib.layers.batch_norm(X1)
    Y = tf.placeholder(tf.float32, [None, 1])  # (?, )

    conv_layer_1= tf.layers.conv3d(inputs=X, filters=32,
                                  kernel_size=[2, 2, 5],
                                  padding="valid",
                                  strides=[2, 2, 5],
                                  activation=tf.nn.relu)
    print("conv1", conv_layer_1.shape)

    conv_layer_2 = tf.layers.conv3d(inputs=conv_layer_1, filters=64,
                                    kernel_size=[5, 5, 5],
                                    padding="same",
                                    activation=tf.nn.relu)
    print("conv2", conv_layer_2.shape)

    conv_layer_3= tf.layers.conv3d(inputs=conv_layer_2, filters=128,
                                   kernel_size=[1, 1, 5],
                                   padding="valid",
                                   strides=[1, 1, 5],
                                   activation=tf.nn.relu)
    print("conv3", conv_layer_3.shape)

    conv_layer_4 = tf.layers.conv3d(inputs=conv_layer_3, filters=128,
                                    kernel_size=[1, 1, 1],
                                    padding="valid",
                                    strides=[1, 1, 3],
                                    activation=tf.nn.relu)
    print("conv4", conv_layer_4.shape)

    conv_layer_5 = tf.layers.conv3d(inputs=conv_layer_4, filters=128,
                                    kernel_size=[1, 1, 1],
                                    padding="valid",
                                    strides=[1, 1, 3],
                                    activation=tf.nn.relu)
    print("conv5", conv_layer_5.shape)

    post_conv = tf.reshape(conv_layer_5, [-1, 4, 5, 128])
    print("post_conv", post_conv.shape)

    pooling_layer = tf.layers.max_pooling2d(inputs=post_conv, pool_size=[2, 2], strides=2)
    print("pooling layer", pooling_layer.shape)

    flat_layer = tf.reshape(pooling_layer, [-1, 2 * 2 * 128])
    print("flat layer", flat_layer.shape)

    dense_layer = tf.layers.dense(inputs=flat_layer, units=1024, activation=tf.nn.relu)
    print("dense layer", dense_layer.shape)

    dropout_layer = tf.nn.dropout(dense_layer, keep_prob=0.8)

    logit_layer = tf.layers.dense(dropout_layer, units=1)
    print("logit layer", logit_layer.shape)


    cost = tf.losses.mean_squared_error(Y, logit_layer)
    #cost = tf.reduce_mean(tf.square(logit_layer - Y))
    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

    # parameters
    epoch_size = 50
    batch_size = 100

    with tf.Session() as sess:
        # -- Initialize
        init = tf.global_variables_initializer()
        sess.run(init)
        print("Learning Start")
        batch_x, batch_y = get_batch_data(x_train, y_train, batch_size)
        for epoch in range(epoch_size):
            avg_cost = 0
            for i in range(len(batch_x)):
                c, hy, _ = sess.run([cost, logit_layer, optimizer], feed_dict={X: batch_x[i], Y: batch_y[i]})
                avg_cost += c
            avg_cost = avg_cost / len(batch_x)
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
        # -- Eval
        c, hy, _ = sess.run([cost, logit_layer, optimizer], feed_dict={X: x_test, Y: y_test})

        prd = np.array(hy).squeeze()
        cal = np.array(y_test).squeeze()

        min_val = min(prd.min(), cal.min())
        max_val = max(prd.max(), cal.max())
        import matplotlib.pyplot as plt
        plt.scatter(prd, cal)
        plt.plot([min_val, max_val], [min_val, max_val], color='r')
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
        plt.show()


