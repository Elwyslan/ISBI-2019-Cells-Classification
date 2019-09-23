from keras.layers import Dropout, Dense, LeakyReLU, Activation
from keras import optimizers, regularizers
from keras.models import Sequential
from keras.callbacks import TensorBoard, Callback, LearningRateScheduler
from keras import backend as K
from pathlib import Path
from sklearn.metrics import confusion_matrix
from collections import Counter
import time
import copy
import os
import shutil
import numpy as np
import pandas as pd

augmPatLvDiv_TRAIN = Path('data/AugmPatLvDiv_TRAIN-TEXTUREData_75-Features_20000-images.csv')
augmPatLvDiv_VALIDATION = Path('data/AugmPatLvDiv_VALIDATION-TEXTUREData_75-Features_5000-images.csv')
augmRndDiv_TRAIN = Path('data/AugmRndDiv_TRAIN-TEXTUREData_75-Features_20000-images.csv')
augmRndDiv_VALIDATION = Path('data/AugmRndDiv_VALIDATION-TEXTUREData_75-Features_5000-images.csv')
patLvDiv_TEST = Path('data/PatLvDiv_TEST-TEXTUREData_75-Features_550-images.csv')
patLvDiv_TRAIN = Path('data/PatLvDiv_TRAIN-TEXTUREData_75-Features_7224-images.csv')
patLvDiv_VALIDATION = Path('data/PatLvDiv_VALIDATION-TEXTUREData_75-Features_2887-images.csv')
rndDiv_TEST = Path('data/rndDiv_TEST-TEXTUREData_75-Features_1066-images.csv')
rndDiv_TRAIN = Path('data/rndDiv_TRAIN-TEXTUREData_75-Features_6398-images.csv')
rndDiv_VALIDATION = Path('data/rndDiv_VALIDATION-TEXTUREData_75-Features_3197-images.csv')

ALL_LABEL = [1.0, 0.0]
HEM_LABEL = [0.0, 1.0]

#Hyperparameters
MAX_EPOCHS = 150
BATCH_SIZE = 500
DROPOUT_RATE = 0.5
LEAKYRELU_RATE = 0.005
L1L2_REG = 0.00005
LR_AT_EPOCH0 = 1e-3
LR_AT_MAX_EPOCH = 1e-6

def LR_decay(epoch):
    #Update Learning rate
    decayRate = (1/MAX_EPOCHS)*np.log(LR_AT_MAX_EPOCH/LR_AT_EPOCH0)
    return np.round(LR_AT_EPOCH0 * np.exp(decayRate*epoch), decimals=6)

def evaluateModel(model, x_valid, y_valid):
    falseNegative = 0
    trueNegative = 0
    falsePositive = 0
    truePositive = 0
    y_true = []
    y_pred = []
    for i in range(len(y_valid)):
        #printIntermediateOutputs(model, x_valid[i,:].reshape(1,-1))
        #Do predictions
        predictions = model.predict(x_valid[i,:].reshape(1,-1))
        #Check predictions - Test for ALL cells
        if predictions[0][0] > predictions[0][1]:#Model Predict ALL...
            y_pred.append('ALL')
            if y_valid[i][0] == 1.0:#... and cell type is ALL => True Positive
                truePositive += 1
            elif y_valid[i][1] == 1.0:#... and cell type is HEM => False Positive
                falsePositive += 1
        elif predictions[0][1] > predictions[0][0]:#Model Predict HEM...
            y_pred.append('HEM')
            if y_valid[i][0] == 1.0:#... and cell type is ALL => False Negative
                falseNegative += 1
            elif y_valid[i][1] == 1.0:#... and cell type is HEM => True Negative
                trueNegative += 1
        
        if y_valid[i][0] == 1.0:
            y_true.append('ALL')
        elif y_valid[i][1] == 1.0:
            y_true.append('HEM')

    #Epsilon avoid division by zero
    epsilon = 1e-9
    #sensitivity/recall =  measures the proportion of actual positives that are correctly identified as such
    sensitivity = truePositive/(truePositive+falseNegative+epsilon)
    #specificity = measures the proportion of actual negatives that are correctly identified as such
    specificity = trueNegative/(trueNegative+falsePositive+epsilon)
    #accuracy = measures the systematic error
    accuracy = (truePositive+trueNegative)/(truePositive + trueNegative + falsePositive + falseNegative+epsilon)
    #precision = description of random errors, a measure of statistical variability.
    precision = truePositive/(truePositive+falsePositive+epsilon)

    sensitivity = np.round((sensitivity*100.0), decimals=2)
    specificity = np.round((specificity*100.0), decimals=2)
    accuracy = np.round((accuracy*100.0), decimals=2)
    precision = np.round((precision*100.0), decimals=2)

    print('###################')
    print(f'Sensitivity: {sensitivity}%')
    print(f'Specificity: {specificity}%')
    print(f'Accuracy: {accuracy}%')
    print(f'Precision: {precision}%')
    print('###################')
    return sensitivity, specificity, accuracy, precision


def balanceDataset(df):
    targets = df['cellType(ALL=1, HEM=-1)']
    cnt = dict(Counter(targets))
    if cnt[-1.0] == cnt[1.0]:
        return df
    ALLtargets = targets.where(targets==1)
    ALLtargets.dropna(axis=0, how='any', inplace=True)
    ALLtargets = ALLtargets.index.tolist()
    HEMtargets = targets.where(targets==-1)
    HEMtargets.dropna(axis=0, how='any', inplace=True)
    HEMtargets = HEMtargets.index.tolist()
    np.random.shuffle(ALLtargets)
    np.random.shuffle(HEMtargets)
    if len(ALLtargets) > len(HEMtargets):
        df = df.drop(ALLtargets[0:(len(ALLtargets) - len(HEMtargets))])
    elif len(HEMtargets) > len(ALLtargets):
        df = df.drop(HEMtargets[0:(len(HEMtargets) - len(ALLtargets))])
    return df

def prepareData(train_df, valid_df):
    #Prepare Validation data
    y_valid = list(valid_df['cellType(ALL=1, HEM=-1)'].values)
    for i in range(len(y_valid)):
        if y_valid[i]==-1:
            y_valid[i] = HEM_LABEL
        elif y_valid[i]==1:
            y_valid[i] = ALL_LABEL
    y_valid = np.array(y_valid)
    x_valid = valid_df.drop(['cellType(ALL=1, HEM=-1)'], axis=1)
    for col in x_valid.columns:
        x_valid[col] = (x_valid[col] - train_df[col].mean()) / train_df[col].std() #mean=0, std=1
    x_valid = x_valid.values

    #Prepare Train data
    y_train = list(train_df['cellType(ALL=1, HEM=-1)'].values)
    for i in range(len(y_train)):
        if y_train[i]==-1:
            y_train[i] = HEM_LABEL
        elif y_train[i]==1:
            y_train[i] = ALL_LABEL
    y_train = np.array(y_train)
    x_train = train_df.drop(['cellType(ALL=1, HEM=-1)'], axis=1)
    for col in x_train.columns:
        x_train[col] = (x_train[col] - train_df[col].mean()) / train_df[col].std() #mean=0, std=1
    x_train = x_train.values
    return x_train, y_train, x_valid, y_valid

def evaluateNeuralNet(train_df, valid_df, prefix):
    x_train, y_train, x_valid, y_valid = prepareData(train_df, valid_df)
    for activationFunction in actFuncs:
        for nH in noOfHiddenLayers:
            for nNeu in noOfNeuro:
                model = Sequential()
                for ly in range(nH):
                    if ly==0:
                        model.add(Dense(nNeu, input_shape=(x_train.shape[1],)))
                    else:
                        model.add(Dense(nNeu))
                    model.add(Activation(activationFunction))
                    model.add(Dropout(0.5))
                model.add(Dense(2, activation='softmax'))
                opt = optimizers.Adam(lr=LR_AT_EPOCH0, decay=0.0 ,beta_1=0.9, beta_2=0.999, epsilon=1e-8, amsgrad=False)
                model.compile(loss='categorical_crossentropy',
                              optimizer=opt,
                              metrics=['accuracy'])
                print(model.summary())
                LOG_DIR = f'results/NeuralNetResults/{prefix}_HL-{nH}_NEU-{nNeu}-Acti-{activationFunction}/'
                if not os.path.exists(Path(LOG_DIR)):
                    os.mkdir(Path(LOG_DIR))
                else:
                    shutil.rmtree(Path(LOG_DIR))
                    os.mkdir(Path(LOG_DIR))
                lrSched = LearningRateScheduler(LR_decay, verbose=1)
                model.fit(x_train, y_train,
                          batch_size=BATCH_SIZE,
                          epochs=MAX_EPOCHS,
                          validation_data=(x_valid, y_valid),
                          callbacks=[TensorBoard(log_dir=LOG_DIR),
                                     lrSched],
                          shuffle=True)
                #sens, spec, acc, prec = evaluateModel(model, x_valid, y_valid)
                del model
                K.clear_session()

if __name__ == '__main__':
    noOfHiddenLayers = [l for l in range(2,5)]
    noOfNeuro = [32, 128, 512, 1024]
    actFuncs = ['tanh', 'relu', 'sigmoid']
    
    augmPatLvDiv_TRAIN = pd.read_csv(augmPatLvDiv_TRAIN, index_col=0)
    augmPatLvDiv_VALIDATION = pd.read_csv(augmPatLvDiv_VALIDATION, index_col=0)
    print('augmPatLvDiv')
    print(Counter(augmPatLvDiv_TRAIN['cellType(ALL=1, HEM=-1)']))
    print(Counter(augmPatLvDiv_VALIDATION['cellType(ALL=1, HEM=-1)']))
    #evaluateNeuralNet(augmPatLvDiv_TRAIN, augmPatLvDiv_VALIDATION, 'augmPatLvDiv')
    
    augmRndDiv_TRAIN = pd.read_csv(augmRndDiv_TRAIN, index_col=0)
    augmRndDiv_VALIDATION = pd.read_csv(augmRndDiv_VALIDATION, index_col=0)
    print('augmRndDiv')
    print(Counter(augmRndDiv_TRAIN['cellType(ALL=1, HEM=-1)']))
    print(Counter(augmRndDiv_VALIDATION['cellType(ALL=1, HEM=-1)']))
    #evaluateNeuralNet(augmRndDiv_TRAIN, augmRndDiv_VALIDATION, 'augmRndDiv')

    patLvDiv_TRAIN = pd.read_csv(patLvDiv_TRAIN, index_col=0)
    patLvDiv_VALIDATION = pd.read_csv(patLvDiv_VALIDATION, index_col=0)
    patLvDiv_TRAIN = balanceDataset(patLvDiv_TRAIN)
    patLvDiv_VALIDATION = balanceDataset(patLvDiv_VALIDATION)
    print('patLvDiv')
    print(Counter(patLvDiv_TRAIN['cellType(ALL=1, HEM=-1)']))
    print(Counter(patLvDiv_VALIDATION['cellType(ALL=1, HEM=-1)']))
    #evaluateNeuralNet(patLvDiv_TRAIN, patLvDiv_VALIDATION, 'patLvDiv')

    rndDiv_TRAIN = pd.read_csv(rndDiv_TRAIN, index_col=0)
    rndDiv_VALIDATION = pd.read_csv(rndDiv_VALIDATION, index_col=0)
    rndDiv_TRAIN = balanceDataset(rndDiv_TRAIN)
    rndDiv_VALIDATION = balanceDataset(rndDiv_VALIDATION)
    print('rndDiv')
    print(Counter(rndDiv_TRAIN['cellType(ALL=1, HEM=-1)']))
    print(Counter(rndDiv_VALIDATION['cellType(ALL=1, HEM=-1)']))
    #evaluateNeuralNet(rndDiv_TRAIN, rndDiv_VALIDATION, 'rndDiv')
    
    #print(f"\nEnd Script!\n{'#'*50}")
    #quit()

    patLvDiv_TEST = pd.read_csv(patLvDiv_TEST, index_col=0)
    rndDiv_TEST = pd.read_csv(rndDiv_TEST, index_col=0)

    def validPerfAugmPatLvDivImages(train_df, valid_df, test=False):
        #NN_augmPatLvDiv_HL-4_NEU-32-Acti-relu
        x_train, y_train, x_valid, y_valid = prepareData(train_df, valid_df)
        model = Sequential()
        model.add(Dense(128,activation='relu',input_shape=(x_train.shape[1],)))
        model.add(Dropout(0.5))
        model.add(Dense(128,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(128,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))
        opt = optimizers.Adam(lr=LR_AT_EPOCH0, decay=0.0 ,beta_1=0.9, beta_2=0.999, epsilon=1e-8, amsgrad=False)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
        print(model.summary())
        LOG_DIR = f'results/NeuralNetResults/NN_augmPatLvDiv/'
        if not os.path.exists(Path(LOG_DIR)):
            os.mkdir(Path(LOG_DIR))
        else:
            shutil.rmtree(Path(LOG_DIR))
            os.mkdir(Path(LOG_DIR))
        lrSched = LearningRateScheduler(LR_decay, verbose=1)
        model.fit(x_train, y_train,
                  batch_size=BATCH_SIZE,
                  epochs=MAX_EPOCHS,
                  validation_data=(x_valid, y_valid),
                  callbacks=[TensorBoard(log_dir=LOG_DIR),
                             lrSched],
                  shuffle=True)
        if test:
            return evaluateModel(model, x_valid, y_valid)
        del model
        K.clear_session()
        
    def validPerfAugmRndDivImages(train_df, valid_df, test=False):
        #NN_augmRndDiv_HL-3_NEU-512-Acti-relu
        x_train, y_train, x_valid, y_valid = prepareData(train_df, valid_df)
        model = Sequential()
        model.add(Dense(1024,activation='relu',input_shape=(x_train.shape[1],)))
        model.add(Dropout(0.5))
        model.add(Dense(1024,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))
        opt = optimizers.Adam(lr=LR_AT_EPOCH0, decay=0.0 ,beta_1=0.9, beta_2=0.999, epsilon=1e-8, amsgrad=False)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
        print(model.summary())
        LOG_DIR = f'results/NeuralNetResults/NN_augmRndDiv/'
        if not os.path.exists(Path(LOG_DIR)):
            os.mkdir(Path(LOG_DIR))
        else:
            shutil.rmtree(Path(LOG_DIR))
            os.mkdir(Path(LOG_DIR))
        lrSched = LearningRateScheduler(LR_decay, verbose=1)
        model.fit(x_train, y_train,
                  batch_size=BATCH_SIZE,
                  epochs=MAX_EPOCHS,
                  validation_data=(x_valid, y_valid),
                  callbacks=[TensorBoard(log_dir=LOG_DIR),
                             lrSched],
                  shuffle=True)
        if test:
            return evaluateModel(model, x_valid, y_valid)
        del model
        K.clear_session()

    def validPerfPatLvDivImages(train_df, valid_df, test=False):
        #NN_patLvDiv_HL-2_NEU-128-Acti-relu
        x_train, y_train, x_valid, y_valid = prepareData(train_df, valid_df)
        model = Sequential()
        model.add(Dense(512,activation='relu',input_shape=(x_train.shape[1],)))
        model.add(Dropout(0.5))
        model.add(Dense(512,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(512,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(512,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))
        opt = optimizers.Adam(lr=LR_AT_EPOCH0, decay=0.0 ,beta_1=0.9, beta_2=0.999, epsilon=1e-8, amsgrad=False)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
        print(model.summary())
        LOG_DIR = f'results/NeuralNetResults/NN_patLvDiv/'
        if not os.path.exists(Path(LOG_DIR)):
            os.mkdir(Path(LOG_DIR))
        else:
            shutil.rmtree(Path(LOG_DIR))
            os.mkdir(Path(LOG_DIR))
        lrSched = LearningRateScheduler(LR_decay, verbose=1)
        model.fit(x_train, y_train,
                  batch_size=BATCH_SIZE,
                  epochs=MAX_EPOCHS,
                  validation_data=(x_valid, y_valid),
                  callbacks=[TensorBoard(log_dir=LOG_DIR),
                             lrSched],
                  shuffle=True)
        if test:
            return evaluateModel(model, x_valid, y_valid)
        del model
        K.clear_session()

    def validPerfRndDivImages(train_df, valid_df, test=False):
        #NN_rndDiv_HL-2_NEU-1024-Acti-relu
        x_train, y_train, x_valid, y_valid = prepareData(train_df, valid_df)
        model = Sequential()
        model.add(Dense(1024,activation='relu',input_shape=(x_train.shape[1],)))
        model.add(Dropout(0.5))
        model.add(Dense(1024,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))
        opt = optimizers.Adam(lr=LR_AT_EPOCH0, decay=0.0 ,beta_1=0.9, beta_2=0.999, epsilon=1e-8, amsgrad=False)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
        print(model.summary())
        LOG_DIR = f'results/NeuralNetResults/NN_rndDiv/'
        if not os.path.exists(Path(LOG_DIR)):
            os.mkdir(Path(LOG_DIR))
        else:
            shutil.rmtree(Path(LOG_DIR))
            os.mkdir(Path(LOG_DIR))
        lrSched = LearningRateScheduler(LR_decay, verbose=1)
        model.fit(x_train, y_train,
                  batch_size=BATCH_SIZE,
                  epochs=MAX_EPOCHS,
                  validation_data=(x_valid, y_valid),
                  callbacks=[TensorBoard(log_dir=LOG_DIR),
                             lrSched],
                  shuffle=True)
        if test:
            return evaluateModel(model, x_valid, y_valid)
        del model
        K.clear_session()

    #TEST
    nn_df = pd.DataFrame()
    collectedData = {}
    collectedData['testID'] = 'AugmPatLvDivImages'
    sens, spec, acc, prec = validPerfAugmPatLvDivImages(augmPatLvDiv_TRAIN, balanceDataset(patLvDiv_TEST), test=True)
    collectedData[f'test-patLvDiv_NN_sens'] = sens
    collectedData[f'test-patLvDiv_NN_spec'] = spec
    collectedData[f'test-patLvDiv_NN_acc'] = acc
    collectedData[f'test-patLvDiv_NN_prec'] = prec
    collectedData[f'test-patLvDiv_NN_F1sco'] = np.round(2*((prec*sens)/(prec+sens)), decimals=2)
    collectedData[f'test-patLvDiv_NN_balAcc'] = np.round((sens+spec)/2, decimals=2)
    sens, spec, acc, prec =  validPerfAugmPatLvDivImages(augmPatLvDiv_TRAIN, balanceDataset(rndDiv_TEST), test=True)
    collectedData[f'test-rndDiv_NN_sens'] = sens
    collectedData[f'test-rndDiv_NN_spec'] = spec
    collectedData[f'test-rndDiv_NN_acc'] = acc
    collectedData[f'test-rndDiv_NN_prec'] = prec
    collectedData[f'test-rndDiv_NN_F1sco'] = np.round(2*((prec*sens)/(prec+sens)), decimals=2)
    collectedData[f'test-rndDiv_NN_balAcc'] = np.round((sens+spec)/2, decimals=2)
    nn_df = nn_df.append(collectedData, ignore_index=True)

    ####################################################################################################################

    collectedData = {}
    collectedData['testID'] = 'AugmRndDivImages'
    sens, spec, acc, prec = validPerfAugmRndDivImages(augmRndDiv_TRAIN, balanceDataset(patLvDiv_TEST), test=True)
    collectedData[f'test-patLvDiv_NN_sens'] = sens
    collectedData[f'test-patLvDiv_NN_spec'] = spec
    collectedData[f'test-patLvDiv_NN_acc'] = acc
    collectedData[f'test-patLvDiv_NN_prec'] = prec
    collectedData[f'test-patLvDiv_NN_F1sco'] = np.round(2*((prec*sens)/(prec+sens)), decimals=2)
    collectedData[f'test-patLvDiv_NN_balAcc'] = np.round((sens+spec)/2, decimals=2)
    sens, spec, acc, prec =  validPerfAugmRndDivImages(augmRndDiv_TRAIN, balanceDataset(rndDiv_TEST), test=True)
    collectedData[f'test-rndDiv_NN_sens'] = sens
    collectedData[f'test-rndDiv_NN_spec'] = spec
    collectedData[f'test-rndDiv_NN_acc'] = acc
    collectedData[f'test-rndDiv_NN_prec'] = prec
    collectedData[f'test-rndDiv_NN_F1sco'] = np.round(2*((prec*sens)/(prec+sens)), decimals=2)
    collectedData[f'test-rndDiv_NN_balAcc'] = np.round((sens+spec)/2, decimals=2)
    nn_df = nn_df.append(collectedData, ignore_index=True)

    ####################################################################################################################

    collectedData = {}
    collectedData['testID'] = 'PatLvDivImages'
    sens, spec, acc, prec = validPerfPatLvDivImages(patLvDiv_TRAIN, balanceDataset(patLvDiv_TEST), test=True)
    collectedData[f'test-patLvDiv_NN_sens'] = sens
    collectedData[f'test-patLvDiv_NN_spec'] = spec
    collectedData[f'test-patLvDiv_NN_acc'] = acc
    collectedData[f'test-patLvDiv_NN_prec'] = prec
    collectedData[f'test-patLvDiv_NN_F1sco'] = np.round(2*((prec*sens)/(prec+sens)), decimals=2)
    collectedData[f'test-patLvDiv_NN_balAcc'] = np.round((sens+spec)/2, decimals=2)
    sens, spec, acc, prec = validPerfPatLvDivImages(patLvDiv_TRAIN, balanceDataset(rndDiv_TEST), test=True)
    collectedData[f'test-rndDiv_NN_sens'] = sens
    collectedData[f'test-rndDiv_NN_spec'] = spec
    collectedData[f'test-rndDiv_NN_acc'] = acc
    collectedData[f'test-rndDiv_NN_prec'] = prec
    collectedData[f'test-rndDiv_NN_F1sco'] = np.round(2*((prec*sens)/(prec+sens)), decimals=2)
    collectedData[f'test-rndDiv_NN_balAcc'] = np.round((sens+spec)/2, decimals=2)
    nn_df = nn_df.append(collectedData, ignore_index=True)

    ####################################################################################################################

    collectedData = {}
    collectedData['testID'] = 'RndDivImages'
    sens, spec, acc, prec = validPerfRndDivImages(rndDiv_TRAIN, balanceDataset(patLvDiv_TEST), test=True)
    collectedData[f'test-patLvDiv_NN_sens'] = sens
    collectedData[f'test-patLvDiv_NN_spec'] = spec
    collectedData[f'test-patLvDiv_NN_acc'] = acc
    collectedData[f'test-patLvDiv_NN_prec'] = prec
    collectedData[f'test-patLvDiv_NN_F1sco'] = np.round(2*((prec*sens)/(prec+sens)), decimals=2)
    collectedData[f'test-patLvDiv_NN_balAcc'] = np.round((sens+spec)/2, decimals=2)
    sens, spec, acc, prec = validPerfRndDivImages(rndDiv_TRAIN, balanceDataset(rndDiv_TEST), test=True)
    collectedData[f'test-rndDiv_NN_sens'] = sens
    collectedData[f'test-rndDiv_NN_spec'] = spec
    collectedData[f'test-rndDiv_NN_acc'] = acc
    collectedData[f'test-rndDiv_NN_prec'] = prec
    collectedData[f'test-rndDiv_NN_F1sco'] = np.round(2*((prec*sens)/(prec+sens)), decimals=2)
    collectedData[f'test-rndDiv_NN_balAcc'] = np.round((sens+spec)/2, decimals=2)
    nn_df = nn_df.append(collectedData, ignore_index=True)

    cols = nn_df.columns.tolist()
    cols.remove('testID')
    nn_df = nn_df[['testID'] + cols]
    nn_df.to_csv(Path('results/NN_TestPerformance_TextureData.csv'))
