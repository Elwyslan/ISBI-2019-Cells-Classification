import numpy as np
import keras
from keras import optimizers, regularizers
from keras.callbacks import TensorBoard, Callback, LearningRateScheduler
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.layers import Input, Dropout, Dense, BatchNormalization,\
                         Activation, MaxPooling2D, Conv2D, Flatten,\
                         GlobalMaxPooling2D
LR_AT_EPOCH0 = 5e-3
LR_AT_MAX_EPOCH = 1e-6
MAX_EPOCHS = 10
def LR_decay(epoch):
    #Update Learning rate
    decayRate = (1/MAX_EPOCHS)*np.log(LR_AT_MAX_EPOCH/LR_AT_EPOCH0)
    return np.round(LR_AT_EPOCH0 * np.exp(decayRate*epoch), decimals=6)

# input image dimensions
img_rows, img_cols = 28, 28

batch_size = 100
num_classes = 10
epochs = 12

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

model.add(Conv2D(input_shape=input_shape, filters=64, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(input_shape=input_shape, filters=64, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#Fully-Connected Layers
model.add(GlobalMaxPooling2D())
model.add(Dense(7, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

#opt = optimizers.Adam(lr=LR_AT_EPOCH0, decay=0.0 ,beta_1=0.9, beta_2=0.999, epsilon=1e-8, amsgrad=False)
opt = optimizers.Adadelta()
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

lrSched = LearningRateScheduler(LR_decay, verbose=1)
LOG_DIR = 'logdir/'
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=MAX_EPOCHS,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[TensorBoard(log_dir=LOG_DIR)])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
    
print(f"\nEnd Script!\n{'#'*50}")

"""
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
"""