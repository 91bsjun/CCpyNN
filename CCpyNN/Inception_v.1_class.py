import numpy as np
import tensorflow as tf
from pymatgen.core.structure import IStructure
from CCpyNN.StructureEncoder import structure_encoder

class Model:
    def __init__(self, sess, name, output_dim, activation_fn=tf.nn.relu,
                 loss_fn=tf.losses.mean_squared_error, optimizer=tf.train.AdamOptimizer, learning_rate=0.001):
        self.sess = sess
        self.name = name
        self.output_dim = output_dim
        with tf.variable_scope(self.name):
            self.X = tf.placeholder(tf.float32, [None, None, 10, 225], name='X')  # (?, N, 10, 225)
            self.Y = tf.placeholder(tf.float32, [None, 1], name='Y')  # (?, 1)
            self.keep_prob = tf.placeholder(tf.float32)
            self.mode = tf.placeholder(tf.bool, name='mode')

            input_layer = self.X
            stem_layer = self.stem(input_layer)
            deception_layer = self.deception(stem_layer)
            logit_layer = self.kick(deception_layer)

            self.loss = loss_fn(self.Y, logit_layer)
            self.loss = tf.reduce_mean(self.loss, name='loss')

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=name)
            with tf.control_dependencies(update_ops):
                self.optimzer = optimizer(learning_rate=learning_rate).minimize(self.loss)

    def stem(self, input_layer):
        fcX = tf.layers.dense(input_layer, units=128)
        bnX = tf.layers.batch_normalization(fcX)
        expX = tf.expand_dims(bnX, axis=4)

        return expX

    def deception(self, input_layer):
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

    def kick(self, input_layer):
        flat_layer = tf.layers.flatten(input_layer)
        dense_layer = tf.layers.dense(inputs=flat_layer, units=1024)
        dropout_layer = tf.nn.dropout(dense_layer, keep_prob=self.keep_prob)
        logit_layer = tf.layers.dense(dropout_layer, units=self.output_dim)

        return logit_layer

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