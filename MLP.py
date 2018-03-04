from __future__ import print_function

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import time

start_time = time.time()
batch_size = 128
num_classes = 1
epochs = 100

# the data, shuffled and split between train and test sets
df = (pd.read_csv('data.csv'))
train, test = train_test_split(df, test_size = 0.2, random_state = None, shuffle = False, stratify = None)

x_train = train.iloc[:,0:2]
y_train = train.iloc[:,-1]
x_test = train.iloc[:,0:2]
y_test = train.iloc[:,-1]

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


model = Sequential()
model.add(Dense(2, activation='relu', input_shape=(2,)))
model.add(Dropout(0.2))
model.add(Dense(2, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=SGD(lr = 0.01, momentum = 0.5),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print("--- %s seconds ---" % (time.time() - start_time))
