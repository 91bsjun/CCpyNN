import numpy as np
import tensorflow as tf
import pandas as pd
try:
    from CCpyNN.HierarchicalCrystal import StructureToMatrixEncoder
except:
    from HierarchicalCrystal import StructureToMatrixEncoder

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

class Model:
    def __init__(self, name, output_dim, activation_fn=tf.nn.relu,
                 loss_fn=tf.losses.mean_squared_error, optimize_fn=tf.train.AdamOptimizer,
                 learning_rate=0.001, keep_prob=0.8):
        # self.sess = sess
        self.name = name
        self.output_dim = output_dim
        self.loss_fn = loss_fn
        self.optimize_fn = optimize_fn
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.mode = tf.placeholder(tf.bool, name='mode')
        with tf.variable_scope(self.name):
            self.Y = tf.placeholder(tf.float32, [None, 1], name='Y')  # (?, 1)
            self.initializer(10, 10, 10)



    def initializer(self, cell_a, cell_b, cell_c):
        self.X = tf.placeholder(tf.float32, [None, cell_a, cell_b, cell_c, 92])
        self.rsX = tf.reshape(self.X, [-1, cell_a * cell_b * cell_c, 92])
        input_layer = self.rsX
        stem_layer = self.stem(input_layer)
        deception_layer = self.deception(stem_layer)
        logit_layer = self.kick(deception_layer)

        self.loss = self.loss_fn(self.Y, logit_layer)
        self.loss = tf.reduce_mean(self.loss, name='loss')

        self.optimizer = self.optimize_fn(learning_rate=self.learning_rate).minimize(self.loss)

    def stem(self, input_layer):
        fcX = tf.layers.dense(input_layer, units=128)
        bnX = tf.layers.batch_normalization(fcX)
        expX = tf.expand_dims(bnX, axis=3)

        return expX

    def deception(self, input_layer):
        layer_1 = tf.layers.conv2d(inputs=input_layer, filters=32,
                                   kernel_size=[2, 2],
                                   padding="valid",
                                   strides=[2, 2])

        layer_2 = tf.layers.conv2d(inputs=layer_1, filters=64,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   strides=[2, 2])

        layer_3 = tf.layers.conv2d(inputs=layer_2, filters=128,
                                   kernel_size=[2, 2],
                                   padding="valid",
                                   strides=[2, 2])

        layer_4 = tf.layers.conv2d(inputs=layer_3, filters=128,
                                   kernel_size=[2, 2],
                                   padding="valid",
                                   strides=[2, 2])

        layer_5 = tf.layers.conv2d(inputs=layer_4, filters=128,
                                   kernel_size=[2, 2],
                                   padding="valid",
                                   strides=[2, 2])


        pooling_layer = tf.layers.max_pooling2d(inputs=layer_5, pool_size=[2, 2], strides=1)

        return pooling_layer

    def kick(self, input_layer):
        flat_layer = tf.layers.flatten(input_layer)
        dense_layer = tf.layers.dense(inputs=flat_layer, units=1024)
        dropout_layer = tf.nn.dropout(dense_layer, keep_prob=self.keep_prob)
        logit_layer = tf.layers.dense(dropout_layer, units=self.output_dim)

        return logit_layer

class Solver:
    def __init__(self, sess, model):
        self.model = model
        self.sess = sess

    def train(self, X, Y):
        feed = {
            self.model.X: X,
            self.model.Y: Y,
            self.model.mode: True
        }
        optimizer = self.model.optimizer
        loss = self.model.loss

        return self.sess.run([optimizer, loss], feed_dict=feed)



if __name__ == "__main__":
    sample_size = 500
    epoch_size = 30

    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    model = Model("test", 1)
    solver = Solver(sess, model)

    x_train, y_train, x_test, y_test = get_data(normalize_y=False, sample_size=sample_size)
    print("Learning Start")
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(epoch_size):
        avg_cost = 0
        total_len = 0
        # for key in x_train.keys():
        mini_x = np.array(x_train[(4, 4, 8, 92)], dtype='float32')
        mini_y = np.array(y_train[(4, 4, 8, 92)], dtype='float32')
        cell_a, cell_b, cell_c = 4, 4, 8
        model.initializer(cell_a, cell_b, cell_c)
        _, loss = solver.train(mini_x, mini_y)
        print(loss)