from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import keras
from keras.metrics import binary_accuracy
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import backend as K
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import time

start_time = time.time()
batch_size = 1024
epochs = 1000

# the data, shuffled and split between train and test sets
df = (pd.read_csv('data.csv'))
train, test = train_test_split(df, test_size = 0.2)

x_train = train.iloc[:,0:2]
y_train = train.iloc[:,-1]
x_test = test.iloc[:,0:2]
y_test = test.iloc[:,-1]

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# for idx, value in enumerate(df.ix[:,-1]):
#     if value == 1:
#         marker = "o"
#         col = 'r'
#     else:
#         marker = "x"
#         col = 'b'
#     plt.scatter(df.ix[idx,0], df.ix[idx,1],c = col, marker = marker)
# plt.show()


learning_rates = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,
                    0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,
                    0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,
                    0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,
                    0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,
                    0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,
                    0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,
                    0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,
                    0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,
                    1,1,1,1,1,1,1,1,1,1]
momentum_terms = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,
                    0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,
                    0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,
                    0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,
                    0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,
                    0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,
                    0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,
                    0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,
                    0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,
                    0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
train_accuracies = np.zeros([100])
test_accuracies = np.zeros([100])

for idx, lr in enumerate(learning_rates):
    model = Sequential()
    model.add(Dense(8, activation='sigmoid', input_shape = (2,)))
    model.add(Dense(8, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    model.compile(loss='binary_crossentropy',
                  optimizer=SGD(lr = lr, momentum = learning_rates[idx]),
                  metrics=[binary_accuracy])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    train_accuracies[idx] = history.history['binary_accuracy'][-1]
    test_accuracies[idx] =  history.history['val_binary_accuracy'][-1]


x = momentum_terms
y = learning_rates
z = test_accuracies

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('Momentum Term')
ax.set_ylabel('Learning Rate')
ax.set_zlabel('Accuracy')
ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none');
plt.show()
# x = nodes
# y = nodes2
# z = test_accuracies
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ticks = [1,2,4,8,16,32]
# ax.set_xticks(np.log10(ticks))
# ax.set_xticklabels(ticks)
# ax.set_yticks(np.log10(ticks))
# ax.set_yticklabels(ticks)
# ax.set_xlabel('First Layer Nodes')
# ax.set_ylabel('Second Layer Nodes')
# ax.set_zlabel('Accuracy')
# ax.plot_trisurf(np.log10(x), np.log10(y), z, cmap='viridis', edgecolor='none');
# plt.show()

y_pred = model.predict(x_test)
y_pred[y_pred <= 0.5] = 0
y_pred[y_pred > 0.5] = 1
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("tn: ", tn, "fp: ", fp, "fn: ", fn, "tp: ", tp)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
print("--- %s seconds ---" % (time.time() - start_time))
