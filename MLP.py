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
epochs = 1

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

learning_rates = [0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,
                    0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,
                    0.001,0.001,0.001,0.001,0.001,0.001,
                    0.01,0.01,0.01,0.01,0.01,0.01,
                    0.1,0.1,0.1,0.1,0.1,0.1,
                    1,1,1,1,1,1]
momentum_terms = [0.00001,0.0001,0.001,0.01,0.1,1,
                    0.00001,0.0001,0.001,0.01,0.1,1,
                    0.00001,0.0001,0.001,0.01,0.1,1,
                    0.00001,0.0001,0.001,0.01,0.1,1,
                    0.00001,0.0001,0.001,0.01,0.1,1,
                    0.00001,0.0001,0.001,0.01,0.1,1,]
train_accuracies = np.zeros([36])
test_accuracies = np.zeros([36])
for idx, learn_rate in enumerate(learning_rates):
    model = Sequential()
    model.add(Dense(18, activation='relu', input_shape = (2,)))
    model.add(Dense(18, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    model.compile(loss='binary_crossentropy',
                  optimizer=SGD(lr = learning_rates[idx], momentum = momentum_terms[idx]),
                  metrics=[binary_accuracy])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print(history.history['val_binary_accuracy'][-1])
    train_accuracies[idx] = history.history['binary_accuracy'][-1]
    test_accuracies[idx] =  history.history['val_binary_accuracy'][-1]
    print(test_accuracies)


x = momentum_terms
y = learning_rates
z = test_accuracies
print(test_accuracies)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none');
plt.show()

y_pred = model.predict(x_test)
y_pred[y_pred <= 0.5] = 0
y_pred[y_pred > 0.5] = 1
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("tn: ", tn, "fp: ", fp, "fn: ", fn, "tp: ", tp)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
print("--- %s seconds ---" % (time.time() - start_time))
