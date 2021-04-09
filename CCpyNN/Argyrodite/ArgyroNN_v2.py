import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn import preprocessing

# -- original -- #
df = pd.read_csv("./Data/argyrodite/cage2.csv")
df = df[["avg-size", "std", "vol", "S"]]
dataset = df.copy()

sns.pairplot(df[["avg-size", "std", "vol", "S"]], diag_kind="kde")
plt.show()

train_dataset = dataset.sample(frac=0.90, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_stats = train_dataset.describe()
train_stats.pop("S")
train_stats = train_stats.transpose()

train_labels = train_dataset.pop('S')
test_labels = test_dataset.pop('S')

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)
# -- original end -- #

# -- test data -- #
'''
df = pd.read_csv("./Data/argyrodite/cage2.csv")
df = df[["avg-size", "std", "vol", "S"]]
dataset1 = df.copy()
df = pd.read_csv("./Data/argyrodite/cage2_test.csv")
df = df[["avg-size", "std", "vol", "S"]]
dataset2 = df.copy()

# sns.pairplot(df[["avg-size", "std", "vol", "S"]], diag_kind="kde")
# plt.show()

train_dataset = dataset1
test_dataset = dataset2

train_stats = train_dataset.describe()
train_stats.pop("S")
train_stats = train_stats.transpose()

train_labels = train_dataset.pop('S')
test_labels = test_dataset.pop('S')

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)
'''
# -- test data end -- #


def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model


model = build_model()
model.summary()

example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)

class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000

# patience 매개변수는 성능 향상을 체크할 에포크 횟수입니다
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data, train_labels,
                    epochs=EPOCHS, validation_split = 0.2, verbose=0,
                    callbacks=[early_stop, PrintDot()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure(figsize=(8,12))

  plt.subplot(2,1,1)
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
  #plt.ylim([0,5])
  plt.legend()

  plt.subplot(2,1,2)
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
  #plt.ylim([0,20])
  plt.legend()
  plt.show()

plot_history(history)
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)


print("테스트 세트의 평균 절대 오차: {:5.2f} mS/cm".format(mae))

# test_predictions = model.predict(normed_test_data).flatten()
test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions, color='#0054FF')
plt.xlabel('True Values', fontsize=16)
plt.ylabel('Predictions', fontsize=16)
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
plt.plot([-100, 100], [-100, 100], color='#0054FF')
plt.tight_layout()
plt.show()

trues = []
predicts = []
print('True', 'Prediction')
for i, val in enumerate(test_labels):
    print('%.2f' % val, '%.2f' % test_predictions[i])
    trues.append(val)
    predicts.append(test_predictions[i])
trues = np.array(trues)
predicts = np.array(predicts)

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

trues = [9.1, 3.94, 2.28, 3.27, 3.24]
predicts = [7.42, 9.25, 2.29, 3.24, 2.62]
print("MAPE: ", mean_absolute_percentage_error(trues, predicts))
print(np.abs((trues - predicts) / trues).mean(axis=0) * 100)




df = pd.read_csv("./Data/argyrodite/cage2_test.csv")
df = df[["avg-size", "std", "vol", "S"]]
dataset2 = df.copy()
test_labels = dataset2.pop('S')
normed_test_data = norm(dataset2)

test_predictions = model.predict(normed_test_data).flatten()

print('True', 'Prediction')
for i, val in enumerate(test_labels):
    print('%.2f' % val, '%.2f' % test_predictions[i])