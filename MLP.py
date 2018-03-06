from __future__ import print_function

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import time

start_time = time.time()
batch_size = 128
epochs = 100

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
#     else:
#         marker = "x"
#     plt.scatter(df.ix[idx,0], df.ix[idx,1],c = 'b', marker = marker)
# plt.show()

model = Sequential()
model.add(Dense(124, activation='relu', input_shape = (2,)))
model.add(Dropout(0.5))
model.add(Dense(124, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=SGD(lr = 1, momentum = 0.03),
              metrics=['acc'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)

y_pred = model.predict(x_test)
y_pred[y_pred <= 0.5] = 0
y_pred[y_pred > 0.5] = 1
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("tn: ", tn, "fp: ", fp, "fn: ", fn, "tp: ", tp)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
print("--- %s seconds ---" % (time.time() - start_time))
