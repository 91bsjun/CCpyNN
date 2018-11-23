import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.slim as slim

from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import IStructure

from StructureEncoder import structure_encoder

np.set_printoptions(threshold=np.nan)
pd.set_option('display.max_rows', None)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def conv2d(tempX, tempW):
    return tf.nn.conv2d(tempX, tempW, strides=[1, 2, 2, 1], padding='SAME')


def conv2d_s1(tempX, tempW):
    return tf.nn.conv2d(tempX, tempW, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_3x3_s1(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                          strides=[1, 1, 1, 1], padding='SAME')


def avg_pool_3x3_s1(x):
    return tf.nn.avg_pool(x, ksize=[1, 3, 3, 1],
                          strides=[1, 1, 1, 1], padding='SAME')


if __name__ == "__main__":
    radius = 3
    max_neighbor_num = 10
    training = 0.9
    test = 0.1

    df = pd.read_csv("./Data/metal-alloy-db.v1/00Total_DB.csv")
    # df = df.sample(n=len(df))
    df = df.sample(n=20)

    cifs = "./Data/metal-alloy-db.v1/" + df['DBname'] + ".cif"
    structures = [IStructure.from_file(cif) for cif in cifs]

    print("Encoding structures")
    """
    import pickle
    if "encoded_structure.pkl" in os.listdir("./"):
        print("using pickle")
        with open("encoded_structure.pkl", "rb") as mydata:
            encoded_structures = pickle.load(mydata)
    else:
        encoded_structures = [structure_encoder(structure, radius, max_neighbor_num) for structure in structures]
        with open("encoded_structure.pkl", "wb") as savedata:
            pickle.dump(encoded_structures, savedata)
    """
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
    x_test = x_data[train:train + test]
    y_test = y_data[train:train + test]

    X = tf.placeholder(tf.float32, [None, None, 10, 225])  # (?, N, 10, 225)
    Y = tf.placeholder(tf.float32, [None, 1])  # (?, 1)


def stem(input_layer):
    # ---- step 1 : 7x7
    W1 = weight_variable([7, 7, 225, 64])
    L1 = conv2d(input_layer, W1)

    # ---- step 2 : pool 3x3
    L2 = max_pool_3x3_s1(L1)

    # ---- step 3 : 1x1
    W3 = weight_variable([1, 1, 64, 64])
    L3 = conv2d_s1(L2, W3)  # (?, ?, 5, 64)
    L3 = tf.nn.relu(L3)

    return L3

def dream1(input_layer):
    conv_layer_1= tf.layers.conv3d(inputs=input_layer, filters=32,
                                   kernel_size=[2, 2, 3],
                                   padding="same",
                                   strides=[2, 2, 5],
                                   activation=tf.nn.relu)
    conv_layer_1 = tf.layers.conv3d(inputs=input_layer, filters=32,
                                    kernel_size=[2, 2, 3],
                                    padding="same",
                                    strides=[2, 2, 5],
                                    activation=tf.nn.relu)

def inception_A(input_layer):
    # ---- Inception 0 : 1x1
    W0 = weight_variable([1, 1, 64, 64])
    L0 = conv2d_s1(input_layer, W0)  # (?, ?, 5, 64)

    # ---- Inception a : 1x1 -> 3x3
    Wa1 = weight_variable([1, 1, 64, 64])
    La1 = conv2d_s1(input_layer, Wa1)

    Wa2 = weight_variable([3, 3, 64, 128])
    La2 = conv2d_s1(La1, Wa2)  # (?, ?, 5, 128)
    La = La2

    # ---- Inception b : 1x1 -> 5x5
    Wb1 = weight_variable([1, 1, 64, 64])
    Lb1 = conv2d_s1(input_layer, Wb1)

    Wb2 = weight_variable([5, 5, 64, 128])
    Lb2 = conv2d_s1(Lb1, Wb2)  # (?, ?, 5, 128)
    Lb = Lb2

    # ---- Inception c : pool -> 1x1
    Lc1 = max_pool_3x3_s1(input_layer)

    Wc2 = weight_variable([1, 1, 64, 64])
    Lc2 = conv2d_s1(Lc1, Wc2)  # (?, ?, 5 ,64)
    Lc = Lc2

    # -- concat
    Final_A = tf.concat([L0, La, Lb, Lc], axis=3)  # (?, ?, 5, 384)
    w = tf.Variable(tf.constant(1., shape=[3, 3, 225, 384]))
    DeConnvA = tf.nn.conv2d_transpose(Final_A, filter=w, output_shape=tf.shape(X), strides=[1, 2, 2, 1], padding='SAME')

    Final_A = tf.nn.relu(Final_A)

    return DeConnvA, Final_A
