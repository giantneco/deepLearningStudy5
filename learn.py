#/usr/bin/env python

from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.core import Dropout
from keras.optimizers import SGD
from keras import backend as K
from keras.optimizers import Adagrad
from keras.layers.normalization import BatchNormalization

def weight_variable(shape):
    return K.truncated_normal(shape, stddev=0.01)

datasets.fetch_mldata('MNIST original', data_home=',')
mnist = datasets.fetch_mldata('MNIST original', data_home=',')
n = len(mnist.data)

indices = np.random.permutation(range(n))
X = mnist.data[indices]
y = mnist.target[indices]
Y = np.eye(10)[y.astype(int)]

N_train = 20000
N_validation = 4000

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=N_train)
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, train_size=N_validation)

'''
Model Configuration
'''

n_in = len(X[0])
n_hiddens = [200, 200, 200, 200, 200, 200, 200]
n_out = len(Y[0])
p_keep = 0.5
activation = 'relu'
initializer = weight_variable
initializer = 'lecun_uniform'
optimizer = SGD(lr=0.01)
# optimizer = SGD(lr=0.01, momentum=0.9)
optimizer = Adagrad(lr=0.01)

model = Sequential()
for fan_in, fan_out in zip(([n_in] + n_hiddens)[:-1], n_hiddens):
    model.add(Dense(fan_out,
                    input_dim=fan_in,
                    kernel_initializer=initializer))
    model.add(Activation(activation))
    model.add(BatchNormalization())
    model.add(Dropout(p_keep))

model.add(Dense(n_out, kernel_initializer=initializer))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

'''
Model Learning
'''

epochs = 50
batch_size = 200

hist = model.fit(X_train,
                 Y_train,
                 epochs=epochs,
                 batch_size=batch_size,
                 validation_data=(X_validation, Y_validation))

myplot(hist)

from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

hist = model.fit(X_train,
                 Y_train,
                 epochs=epochs,
                 batch_size=batch_size,
                 validation_data=(X_validation, Y_validation),
                 callbacks=[early_stopping])
