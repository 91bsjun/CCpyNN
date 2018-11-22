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


if __name__ == "__main__":
    radius = 3
    max_neighbor_num = 10
    training = 0.9
    test = 0.1

    df = pd.read_csv("./Data/metal-alloy-db.v1/00Total_DB.csv")
    df = df.sample(n=len(df))
    #df = df.sample(n=110)

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
    x_data = np.array(encoded_structures)

    formation_energy = df['FormationEnergy']
    mean = formation_energy.mean()
    std = formation_energy.std()
    norm_form_energy = (df['FormationEnergy'] - mean) / std
    def norm_back(val, mean, std):
        return val * std + mean
    y_data = [[val] for val in norm_form_energy]
    y_data = np.array(y_data)


    total = len(df)
    train = int(float(total) * training)
    test = int(float(total) * test)
    x_train = x_data[:train]
    y_train = y_data[:train]
    x_test = x_data[train:train+test]
    y_test = y_data[train:train+test]

    X = tf.placeholder(tf.float32, [None, None, 10, 225])  # (?, N, 10, 104)
    X1 = slim.fully_connected(X, 128)
    X2 = tf.contrib.layers.batch_norm(X1)
    Y = tf.placeholder(tf.float32, [None, 1])  # (?, )

    #W_conv1 = tf.Variable(tf.random_normal([2, 3, 104, 32], stddev=0.01))  # (2, 3, 104, 32)
    W_conv1 = tf.get_variable("W_conv1", shape=[3, 3, 128, 32], initializer=tf.contrib.layers.xavier_initializer())
    h_conv1 = tf.nn.conv2d(X2, W_conv1, strides=[1, 1, 1, 1], padding='SAME')  # (?, ?, 10, 32)
    h_conv1 = tf.nn.relu(h_conv1)
    h_conv1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    #W_conv2 = tf.Variable(tf.random_normal([2, 2, 32, 64], stddev=0.01))  # (2, 2, 32, 64)
    W_conv2 = tf.get_variable("W_conv2", shape=[3, 3, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
    h_conv2 = tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')  # (?, ?, 10, 64)
    h_conv2 = tf.nn.relu(h_conv2)
    h_conv2 = tf.nn.dropout(h_conv2, keep_prob=0.8)  # (?, ?, 5, 64)
    h_conv2 = tf.nn.softmax(h_conv2)

    W_conv3 = tf.get_variable("W_conv3", shape=[3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
    h_conv3 = tf.nn.conv2d(h_conv2, W_conv3, strides=[1, 1, 1, 1], padding='SAME')  # (?, ?, 10, 64)
    h_conv3 = tf.nn.relu(h_conv3)
    h_conv3 = tf.nn.max_pool(h_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    h_conv3 = tf.nn.dropout(h_conv3, keep_prob=0.8) # (?, ?, 3, 128)


    #w = tf.Variable(tf.constant(1., shape=[1, 1, 104, 128]))
    #DeConnv1 = tf.nn.conv2d_transpose(h_conv3, filter=w, output_shape=tf.shape(X), strides=[1, 1, 1, 1], padding='SAME')

    #final = DeConnv1  # (?, ?, 10, 104)
    #final_w = tf.Variable(tf.random_normal([1, 1, 104, 1], stddev=0.01))
    #final_conv = tf.nn.conv2d(final, final_w, strides=[1, 1, 1, 1], padding='SAME') # (?, ?, 10, 1)

    rsum = tf.reduce_sum(h_conv3, axis=2)
    rsum = tf.reduce_sum(rsum, axis=1)

    W = tf.get_variable("W", shape=[128, 1], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.random_normal([1]), name='bias')

    hypothesis = tf.matmul(rsum, W) + b

    cost = tf.losses.mean_squared_error(Y, hypothesis)
    #cost = tf.reduce_mean(tf.square(hypothesis - Y))
    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

    # parameters
    training_epochs = 30
    batch_size = 100

    with tf.Session() as sess:
        # -- Initialize
        init = tf.global_variables_initializer()
        sess.run(init)
        print("Learning Start")
        for epoch in range(training_epochs):
            print("Epoch: ", epoch)
            c, _ = sess.run([cost, optimizer], feed_dict={X: x_train, Y: y_train})
            print("Cost: ", c)

        # -- Eval
        c, hy, _ = sess.run([cost, hypothesis, optimizer], feed_dict={X: x_test, Y: y_test})

        print(hy[:10])
        print(y_test[:10])

