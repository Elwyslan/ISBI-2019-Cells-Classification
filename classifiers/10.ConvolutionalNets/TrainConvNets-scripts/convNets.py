from keras.layers import Input, Dropout, Dense, BatchNormalization,\
                         Activation, MaxPooling2D, Conv2D, Flatten,\
                         GlobalMaxPooling2D, GlobalAveragePooling2D,\
                         LeakyReLU
from keras import optimizers, regularizers, initializers
from keras.models import Sequential, Model, load_model
from keras.callbacks import TensorBoard, Callback, LearningRateScheduler
from keras.utils.io_utils import HDF5Matrix
from keras.utils.vis_utils import plot_model
from keras.applications.xception import Xception
from keras.applications.mobilenet import MobileNet
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras import backend as K
from collections import Counter
import numpy as np
import os
import pandas as pd


TRAIN_DATA_PATH = 'HDF5data/Train_augmRndDiv_20000imgs+250_250_3.h5'
VALIDATION_DATA_PATH = 'HDF5data/Validation_augmRndDiv_5000imgs+250_250_3.h5'

ALL_LABEL = [1.0, 0.0]
HEM_LABEL = [0.0, 1.0]


def LR_decay(epoch):
    #Update Learning rate
    decayRate = (1/MAX_EPOCHS)*np.log(LR_AT_MAX_EPOCH/LR_AT_EPOCH0)
    return np.round(LR_AT_EPOCH0 * np.exp(decayRate*epoch), decimals=6)

class personalCallback(Callback):
    def __init__(self):
        self.lowest_val_loss = 1000.0
        self.highest_val_acc = 0.0
        self.acc = -1
        self.loss = -1
        return

    def on_train_begin(self, logs={}):
        return
 
    def on_train_end(self, logs={}):
        modelName = 'finalEpoch_{}_ACC-{}_LOS-{}.h5'.format(CHOOSED_MODEL, self.acc, self.loss)
        savepath = 'Models/{}'.format(modelName)
        #self.model.save(str(savepath))
 
    def on_epoch_begin(self, epoch, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):
        self.acc = np.round(logs['val_acc'], decimals=6)
        self.loss = np.round(logs['val_loss'], decimals=6)
        saveModel = False
        if self.acc>self.highest_val_acc:
            self.highest_val_acc = self.acc
            saveModel = True
        if self.loss<self.lowest_val_loss:
            self.lowest_val_loss = self.loss
            saveModel = True
        if saveModel:
            modelName = '{}_ACC-{}_LOS-{}_Epoch-{}.h5'.format(CHOOSED_MODEL, self.acc, self.loss, epoch+1)
            savepath = 'Models/{}'.format(modelName)
            self.model.save(str(savepath))
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        return

if __name__ == '__main__':
    #Instantiating HDF5Matrix for the training set
    x_train = HDF5Matrix(TRAIN_DATA_PATH, 'train_imgs')
    y_train = HDF5Matrix(TRAIN_DATA_PATH, 'train_labels')

    #Instantiating HDF5Matrix for the validation set
    x_valid = HDF5Matrix(VALIDATION_DATA_PATH, 'valid_imgs')
    y_valid = HDF5Matrix(VALIDATION_DATA_PATH, 'valid_labels')

    #Check Dataset
    y_t = list(y_train)
    y_t = [list(el) for el in y_t]
    y_t = ['ALL' if el==[1.0, 0.0] else 'HEM' for el in y_t]
    print(Counter(y_t))
    y_v = list(y_valid)
    y_v = [list(el) for el in y_v]
    y_v = ['ALL' if el==[1.0, 0.0] else 'HEM' for el in y_v]
    print(Counter(y_v))
    #quit()
    #initializers.he_uniform(seed=None), initializers.he_normal(seed=None)
    #initializers.lecun_normal(seed=None), initializers.glorot_uniform(seed=None)
    #initializers.glorot_normal(seed=None), initializers.RandomUniform()
    weightInit = initializers.glorot_uniform()
    kernelReg = regularizers.l1_l2(l1=0.00001, l2=0.00001)
    input_shape = x_train.shape[1:]


    #Base Models: xception, VGG16, VGG19, ResNet50, InceptionV3 
    CHOOSED_MODEL = 'InceptionV3'

    #Build Convolution Network
    if CHOOSED_MODEL=='xception':
        #Hyperparameters
        MAX_EPOCHS = 100
        BATCH_SIZE = 20
        LR_AT_EPOCH0 = 5e-6
        kernelReg = regularizers.l1_l2(l1=0.0007, l2=0.0007)
        model_Xception = Xception(include_top=False, weights=None,
                                  input_tensor=None, input_shape=input_shape,
                                  pooling=None, classes=None)
        x = model_Xception.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(2048, activation='relu', kernel_regularizer=kernelReg)(x)
        x = Dropout(0.5)(x)
        predictions = Dense(2, activation="softmax", kernel_regularizer=kernelReg)(x)
        model = Model(inputs=model_Xception.input, outputs=predictions)

    elif CHOOSED_MODEL=='VGG16':
        MAX_EPOCHS = 100
        BATCH_SIZE = 20
        LR_AT_EPOCH0 = 1e-5
        kernelReg = regularizers.l1_l2(l1=0.0001, l2=0.0001)
        model_vgg16 = VGG16(include_top=False, weights=None, input_tensor=None,
                            input_shape=input_shape, pooling=None, classes=2)
        x = model_vgg16.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu', kernel_regularizer=kernelReg)(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu', kernel_regularizer=kernelReg)(x)
        x = Dropout(0.5)(x)
        predictions = Dense(2, activation="softmax", kernel_regularizer=kernelReg)(x)
        model = Model(inputs=model_vgg16.input, outputs=predictions)

    elif CHOOSED_MODEL=='VGG19':
        MAX_EPOCHS = 100
        BATCH_SIZE = 20
        LR_AT_EPOCH0 = 1e-5
        kernelReg = regularizers.l1_l2(l1=0.0001, l2=0.0001)
        model_vgg19 = VGG19(include_top=False, weights=None, input_tensor=None,
                            input_shape=input_shape, pooling=None, classes=2)
        x = model_vgg19.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu', kernel_regularizer=kernelReg)(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu', kernel_regularizer=kernelReg)(x)
        x = Dropout(0.5)(x)
        predictions = Dense(2, activation="softmax", kernel_regularizer=kernelReg)(x)
        model = Model(inputs=model_vgg19.input, outputs=predictions)

    elif CHOOSED_MODEL=='ResNet50':
        MAX_EPOCHS = 100
        BATCH_SIZE = 20
        LR_AT_EPOCH0 = 5e-7
        kernelReg = regularizers.l1_l2(l1=0.0001, l2=0.0001)
        model_resnet50 = ResNet50(include_top=False, weights=None, input_tensor=None,
                                  input_shape=input_shape, pooling=None, classes=2)
        x = model_resnet50.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu', kernel_regularizer=kernelReg)(x)
        x = Dropout(0.5)(x)
        predictions = Dense(2, activation="softmax", kernel_regularizer=kernelReg)(x)
        model = Model(inputs=model_resnet50.input, outputs=predictions)

    elif CHOOSED_MODEL=='InceptionV3':
        MAX_EPOCHS = 100
        BATCH_SIZE = 20
        LR_AT_EPOCH0 = 5e-7
        kernelReg = regularizers.l1_l2(l1=0.0001, l2=0.0001)
        model_InceptionV3 = InceptionV3(include_top=False, weights=None, input_tensor=None,
                                        input_shape=input_shape, pooling=None, classes=2)
        x = model_InceptionV3.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(2048, activation='relu', kernel_regularizer=kernelReg)(x)
        x = Dropout(0.5)(x)
        predictions = Dense(2, activation="softmax", kernel_regularizer=kernelReg)(x)
        model = Model(inputs=model_InceptionV3.input, outputs=predictions)


    #Gradient Descendent Optimizer
    opt = optimizers.Adam(lr=LR_AT_EPOCH0, decay=0.0 ,beta_1=0.9, beta_2=0.999, epsilon=1e-8, amsgrad=False)
    #opt = optimizers.Adadelta()
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    #Print Final Model
    print(model.summary())
    print('\n\nBase Model Architecture: {}\n\n'.format(CHOOSED_MODEL))
    #quit()
    #Create Log Directory
    LOG_DIR = 'logdir/{}/'.format(CHOOSED_MODEL)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    #Train
    lrSched = LearningRateScheduler(LR_decay, verbose=1)
    model.fit(x=x_train, y=y_train,
              batch_size=BATCH_SIZE,
              validation_data=(x_valid, y_valid),
              epochs=MAX_EPOCHS,
              initial_epoch=0,
              verbose=1,
              callbacks=[TensorBoard(log_dir=LOG_DIR),
                         personalCallback()],
              shuffle='batch')

    print("\nEnd Script!\n{}\n".format('#'*50))


"""
model = Sequential()
    # 1st Convolutional Layer
    model.add(Conv2D(filters=96, input_shape=input_shape, kernel_size=(11,11), strides=(4,4), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    
    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))

    # 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))

    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

    # Passing it to a Fully Connected layer
    #model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    # 1st Fully Connected Layer
    model.add(Dense(1024))
    model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4))

    # 2nd Fully Connected Layer
    model.add(Dense(1024))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))

    # Output Layer
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()
"""



"""
#Create Model
    
    #CONV1
    model = Sequential()
    model.add(Conv2D(input_shape=input_shape, filters=128,
                                              kernel_size=(10,10),
                                              strides=(10,10),
                                              padding='valid',
                                              kernel_initializer=weightInit,
                                              kernel_regularizer=kernelReg))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=256,
                     kernel_size=(5,5),
                     strides=(1,1),
                     padding='same',
                     kernel_initializer=weightInit,
                     kernel_regularizer=kernelReg))
    model.add(MaxPooling2D(pool_size=(2,2),
                           strides=(2,2),
                           padding='valid'))
    model.add(BatchNormalization())

    #CONV2
    model.add(Conv2D(filters=256,
                     kernel_size=(5,5),
                     strides=(1,1),
                     padding='same',
                     kernel_initializer=weightInit,
                     kernel_regularizer=kernelReg))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(5,5),
                           strides=(1,1),
                           padding='same'))
    model.add(BatchNormalization())
    
    #CONV3
    model.add(Conv2D(filters=512,
                     kernel_size=(5,5),
                     strides=(1,1),
                     padding='same',
                     kernel_initializer=weightInit,
                     kernel_regularizer=kernelReg))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    #Fully-Connected Layers
    model.add(GlobalAveragePooling2D())
    #model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    opt = optimizers.Adam(lr=LR_AT_EPOCH0, decay=0.0 ,beta_1=0.9, beta_2=0.999, epsilon=1e-8, amsgrad=False)
    #opt = optimizers.Adadelta()
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    print(model.summary())
    quit()

"""
