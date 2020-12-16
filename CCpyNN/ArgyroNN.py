import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense, Activation
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.optimizers import SGD

from sklearn import preprocessing


def get_data(sample_size=None, normalize=False):
    df = pd.read_csv("./Data/argyrodite/cage.csv")
    #df = df[['std', 'S']]
    sns.pairplot(df[["avg-size", "std", "vol", "S"]], diag_kind="kde")
    # sns.jointplot("std", "S", df[["std", "S"]], kind="kde", space=0, color='red')
    plt.show()
    quit()
    training = 0.8
    test = 0.2
    total = len(df)
    train = int(float(total) * training)
    test = int(float(total) * test)

    #df = df.fillna(-1)
    df = df.sample(n=len(df))
    # df = df.sample(n=sample_size)

    def get_xy(in_df):
        y = in_df['S'].to_numpy()
        y = np.expand_dims(y, axis=1)
        x = in_df.drop('S', axis=1).to_numpy()
        # x = np.expand_dims(x, axis=2)    # use when 1d Conv

        return x, y

    def get_xy_np(xy):
        x = xy[:, 0:-1]
        y = xy[:, [-1]]
        # x = np.expand_dims(x, axis=2)    # use when 1d Conv

        return x, y

    if normalize:
        x = df.values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)

        train_xy = x_scaled[:train]
        test_xy = x_scaled[train:train + test]

        x_train, y_train = get_xy_np(train_xy)
        x_test, y_test = get_xy_np(test_xy)



    else:
        train_df = df[:train]
        test_df = df[train:train + test]

        x_train, y_train = get_xy(train_df)
        x_test, y_test = get_xy(test_df)






    return x_train, y_train, x_test, y_test


def evaluate_model(x_train, y_train, x_test, y_test):
    #print(trainX.shape)
    #print(trainy.shape)
    #print(testX.shape)
    #print(testy.shape)


    verbose, epochs, batch_size = 0, 30, 4
    n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mse'])
    #print(model.summary())

    history = model.fit(x_train, y_train, epochs=epochs, verbose=verbose)
    # evaluate model
    results = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
    print(results)

    predictions = model.predict(x_test)
    print(y_test)
    print(predictions)

    # plot loss during training
    plt.title('Loss / Mean Squared Error')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()


def simple_regression(x_train, y_train, x_test, y_test):
    model = Sequential()
    # model.add(Dense(1, input_dim=10))
    model.add(Dense(1, input_dim=1))
    model.add(Activation('linear'))
    model.summary()

    model.compile(loss='mse',
                  optimizer=SGD(lr=1e-5),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=1000)

    predictions = model.predict(x_train)
    score = model.evaluate(x_train, y_train)

    print('Prediction: \n', predictions)
    print('Cost: ', score[0])

    predictions = model.predict(x_test)
    print(y_test)
    print(predictions)
    plt.scatter(y_test, predictions)
    plt.xlabel('True')
    plt.ylabel('Predictied')
    plt.show()


# load data
x_train, y_train, x_test, y_test = get_data(normalize=True)



# repeat experiment

# evaluate_model(x_train, y_train, x_test, y_test)
simple_regression(x_train, y_train, x_test, y_test)



'''
# shape X = (?, 10), Y = (?, 1)    --> (38, 10)   (38, 1)
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=2))
# model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='softmax'))
model.add(Activation('sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

print(model.summary())
#print(x_train.shape)
#print(y_train.shape)
#quit()

callbacks_list = [
    keras.callbacks.ModelCheckpoint(filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='acc', patience=1)
]

history = model.fit(x_train,
                    y_train,
                    batch_size=10,
                    epochs=20,
                    callbacks=callbacks_list,
                    validation_split=0.2,
                    verbose=1)
'''
