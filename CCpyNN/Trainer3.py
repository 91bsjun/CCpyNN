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
    #df = df.sample(n=len(df))
    df = df.sample(n=5)

    cifs = "./Data/metal-alloy-db.v1/" + df['DBname'] + ".cif"
    structures = [IStructure.from_file(cif) for cif in cifs]

    print("Encoding structures")
    '''
    import pickle
    if "encoded_structure.pkl" in os.listdir("./"):
        print("using pickle")
        with open("encoded_structure.pkl", "rb") as mydata:
            encoded_structures = pickle.load(mydata)
    else:
        encoded_structures = [structure_encoder(structure, radius, max_neighbor_num) for structure in structures]
        with open("encoded_structure.pkl", "wb") as savedata:
            pickle.dump(encoded_structures, savedata)
    '''

    formation_energy = df['FormationEnergy']
    mean = formation_energy.mean()
    std = formation_energy.std()
    norm_form_energy = (df['FormationEnergy'] - mean) / std
    def norm_back(val, mean, std):
        return val * std + mean
    y_data = [[val] for val in norm_form_energy]
    y_data = np.array(y_data)

    print("Done")
    encoded_structures = [structure_encoder(structure, radius, max_neighbor_num) for structure in structures]
    x_atom_fea = []
    x_nbr_fea_idx = []
    x_nbr_fea = []
    for i, each_structure in enumerate(encoded_structures):
        x_atom_fea.append(each_structure[0])
        x_nbr_fea_idx.append(each_structure[1])
        x_nbr_fea.append(each_structure[2])

    x_atom_fea = np.array(x_atom_fea, dtype='float32')
    x_nbr_fea_idx = np.array(x_nbr_fea_idx, dtype='float32')
    x_nbr_fea = np.array(x_nbr_fea, dtype='float32')

    atom_fea = tf.placeholder(tf.float32, [None, None, 92])
    nbr_fea_idx = tf.placeholder(tf.float32, [None, None, 10])
    nbr_fea = tf.placeholder(tf.float32, [None, None, 10, 41])
    atom_nbr_fea = tf.placeholder(tf.float32, [None, None, 10, 64])

    atom_fea = tf.contrib.layers.fully_connected(atom_fea, 64)  # (5, 8, 64)   # (5, 8, 10)

    for i in range(3):

        exp_atom_fea = tf.expand_dims(atom_fea, axis=2)
        exp_atom_fea = tf.tile(exp_atom_fea, [1, 1, 10, 1])
        total_nbr_fea = tf.concat([exp_atom_fea, atom_nbr_fea, nbr_fea], axis=3)
        total_gated_fea = tf.contrib.layers.fully_connected(total_nbr_fea, 128)
        total_gated_fea = tf.contrib.layers.batch_norm(total_gated_fea)
        nbr_filter, nbr_core = tf.split(total_gated_fea, 2, axis=3)
        nbr_filter = tf.nn.sigmoid(nbr_filter)
        nbr_core = tf.nn.softplus(nbr_core)
        nbr_summed = tf.reduce_sum(nbr_filter * nbr_core, axis=2)
        nbr_summed = tf.contrib.layers.batch_norm(nbr_summed)
        out = tf.nn.softplus(atom_fea + nbr_summed)
        atom_fea = out
    crys_fea = tf.reduce_mean(atom_fea, axis=1)
    crys_fea = tf.nn.softplus(crys_fea)
    crys_fea = tf.contrib.layers.fully_connected(crys_fea, 32)
    crys_fea = tf.nn.softplus(crys_fea)
    out = tf.contrib.layers.fully_connected(crys_fea, 1)





    total = len(df)
    train = int(float(total) * training)
    test = int(float(total) * test)
    x_train = x_data[:train]
    y_train = y_data[:train]
    x_test = x_data[train:train+test]
    y_test = y_data[train:train+test]


    X = tf.placeholder(tf.float32, [None, None, 10, 225])  # (?, N, 10, 104)
    X1 = slim.fully_connected(X, 128)
    Y = tf.placeholder(tf.float32, [None, 1])  # (?, )

    #W_conv1 = tf.Variable(tf.random_normal([2, 3, 104, 32], stddev=0.01))  # (2, 3, 104, 32)
    W_conv1 = tf.get_variable("W_conv1", shape=[3, 3, 128, 32], initializer=tf.contrib.layers.xavier_initializer())
    h_conv1 = tf.nn.conv2d(X1, W_conv1, strides=[1, 1, 1, 1], padding='SAME')  # (?, ?, 10, 32)
    h_conv1 = tf.nn.relu(h_conv1)
    h_conv1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    #W_conv2 = tf.Variable(tf.random_normal([2, 2, 32, 64], stddev=0.01))  # (2, 2, 32, 64)
    W_conv2 = tf.get_variable("W_conv2", shape=[3, 3, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
    h_conv2 = tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')  # (?, ?, 10, 64)
    h_conv2 = tf.nn.relu(h_conv2)
    h_conv2 = tf.nn.dropout(h_conv2, keep_prob=0.8)  # (?, ?, 5, 64)

    W_conv3 = tf.get_variable("W_conv3", shape=[3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
    h_conv3 = tf.nn.conv2d(h_conv2, W_conv3, strides=[1, 1, 1, 1], padding='SAME')  # (?, ?, 10, 64)
    h_conv3 = tf.nn.relu(h_conv3)
    h_conv3 = tf.nn.max_pool(h_conv3, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')
    h_conv3 = tf.nn.dropout(h_conv3, keep_prob=0.8) # (?, ?, 3, 128)

    w = tf.Variable(tf.constant(1., shape=[1, 3, 225, 128]))
    DeConnv1 = tf.nn.conv2d_transpose(h_conv3, filter=w, output_shape=tf.shape(X), strides=[1, 2, 2, 1], padding='SAME')



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
        c, hy, _ = sess.run([cost, hypothesis, optimizer], feed_dict={X: x_test, Y: y_test})

        print(hy[:10])
        print(y_test[:10])

