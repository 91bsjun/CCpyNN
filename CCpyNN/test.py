import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.slim as slim

from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import IStructure

from CCpyNN.StructureEncoder import structure_encoder
#np.set_printoptions(threshold=np.nan)
pd.set_option('display.max_rows', None)


if __name__ == "__main__":
    radius = 3
    max_neighbor_num = 10
    training = 0.9
    test = 0.1

    df = pd.read_csv("./Data/metal-alloy-db.v1/00Total_DB.csv")
    #df = df.sample(n=len(df))
    df = df.sample(n=10)

    cifs = "./Data/metal-alloy-db.v1/" + df['DBname'] + ".cif"
    structures = [IStructure.from_file(cif) for cif in cifs]


    encoded_structures = [structure_encoder(structure, radius, max_neighbor_num) for structure in structures]


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
    Y = tf.placeholder(tf.float32, [None, 1])  # (?, )

    X1 = tf.contrib.layers.fully_connected(X, 192)
    print(X1.shape)
    X2 = tf.contrib.layers.batch_norm(X1)
    p1, p2 = tf.split(X2, 2, axis=3)
    p1 = tf.nn.sigmoid(p1)
    p2 = tf.nn.softplus(p2)
    summed = tf.reduce_sum(p1 * p2, axis=2)
    out = tf.contrib.layers.batch_norm(summed)
    out = tf.nn.softplus(out)
    pooled = tf.reduce_mean(out, axis=1)
    pooled = tf.nn.softplus(pooled)
    pooled = slim.fully_connected(pooled, 64)
    pooled = tf.nn.softplus(pooled)
    final_out = slim.fully_connected(pooled, 1)

    hypothesis = final_out


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
            c, hy, _ = sess.run([cost, hypothesis, optimizer], feed_dict={X: x_train, Y: y_train})
            print("Cost: ", c)

        # -- Eval
        c, hy, X1, X2, _ = sess.run([cost, hypothesis, X1, X2, optimizer], feed_dict={X: x_test, Y: y_test})

        print(X1)
        #print(X2)

