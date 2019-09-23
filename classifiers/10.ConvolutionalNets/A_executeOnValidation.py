import os
import cv2
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Model, load_model
from keras import backend as K
import pandas as pd

ALL_LABEL = [[1.0, 0.0]]
HEM_LABEL = [[0.0, 1.0]]

xception_AugmPatDiv_nvidia1070 = Path('00.Xception/Nvidia1070_Data/xception_augmPatLvDiv_ACC-0.8462_LOS-10.127935_Epoch-291.h5')
xception_AugmRndDiv_nvidia1070 = Path('00.Xception/Nvidia1070_Data/xception_augmRndDiv_ACC-0.8996_LOS-0.450646_Epoch-245.h5')
xception_AugmPatDiv = Path('00.Xception/Models/xception_AugmPatDiv_ACC-0.8742_LOS-0.460651_Epoch-28.h5')
xception_AugmRndDiv = Path('00.Xception/Models/xception_AugmRndDiv_ACC-0.899_LOS-0.391107_Epoch-15.h5')

vgg16_AugmPatDiv_nvidia1070 = Path('01.VGG16/Nvidia1070_Data/VGG16_augmPatLvDiv_ACC-0.8934_LOS-0.603212_Epoch-89.h5')
vgg16_AugmRndDiv_nvidia1070 = Path('01.VGG16/Nvidia1070_Data/VGG16_augmRndDiv_ACC-0.9198_LOS-0.381919_Epoch-89.h5')
vgg16_AugmPatDiv = Path('01.VGG16/Models/VGG16_AugmPatDiv_ACC-0.8962_LOS-0.287934_Epoch-16.h5')
vgg16_AugmRndDiv = Path('01.VGG16/Models/VGG16_AugmRndDiv_ACC-0.925_LOS-0.321087_Epoch-79.h5')

vgg19_AugmPatDiv_nvidia1070 = Path('02.VGG19/Nvidia1070_data/VGG19_augmPatLvDiv_ACC-0.8912_LOS-0.622873_Epoch-127.h5')
vgg19_AugmRndDiv_nvidia1070 = Path('02.VGG19/Nvidia1070_data/VGG19_augmRndDiv_ACC-0.9274_LOS-0.616726_Epoch-223.h5')
vgg19_AugmPatDiv = Path('02.VGG19/Models/VGG19_AugmPatDiv_ACC-0.884_LOS-0.488645_Epoch-40.h5')
vgg19_AugmRndDiv = Path('02.VGG19/Models/VGG19_AugmRndDiv_ACC-0.9278_LOS-0.334132_Epoch-66.h5')

resNet50_AugmPatDiv = Path('03.ResNet50/Models/ResNet50_AugmPatDiv_ACC-0.7798_LOS-7.938992_Epoch-13.h5')
resNet50_AugmRndDiv = Path('03.ResNet50/Models/ResNet50_AugmRndDiv_ACC-0.799_LOS-2.819559_Epoch-17.h5')
inceptionV3_AugmPatDiv = Path('04.InceptionV3/Models/InceptionV3_AugmPatDiv_ACC-0.7874_LOS-7.363104_Epoch-24.h5')
inceptionV3_AugmRndDiv = Path('04.InceptionV3/Models/InceptionV3_AugmRndDiv_ACC-0.814_LOS-8.046428_Epoch-10.h5')


def processImg(imgPath, subtractMean, divideStdDev, colorScheme, imgSize):
    #Process RGB Images
    if colorScheme=='rgb':
        #Read Image
        img = cv2.cvtColor(cv2.imread(str(imgPath)), cv2.COLOR_BGR2RGB)
        width, height, _ = img.shape
        #Crop to required size
        w_crop = int(np.floor((width-imgSize[0])/2))
        h_crop = int(np.floor((height-imgSize[1])/2))
        img = img[w_crop:w_crop+imgSize[0], h_crop:h_crop+imgSize[1], :]
        #Convert to Float 32
        img = np.array(img, dtype=np.float32)
        #Rescale
        img[:,:,0] = img[:,:,0]/255.0
        img[:,:,1] = img[:,:,1]/255.0
        img[:,:,2] = img[:,:,2]/255.0
        #Subtract mean
        if subtractMean:
            #print('Mean:',np.mean(img[:,:,0]), np.mean(img[:,:,1]), np.mean(img[:,:,2]))
            img[:,:,0] = img[:,:,0] - np.mean(img[:,:,0])
            img[:,:,1] = img[:,:,1] - np.mean(img[:,:,1])
            img[:,:,2] = img[:,:,2] - np.mean(img[:,:,2])
        #Divide standard deviation
        if divideStdDev:
            #print('std:',np.std(img[:,:,0]), np.std(img[:,:,1]), np.std(img[:,:,2]))
            img[:,:,0] = img[:,:,0] / np.std(img[:,:,0])
            img[:,:,1] = img[:,:,1] / np.std(img[:,:,1])
            img[:,:,2] = img[:,:,2] / np.std(img[:,:,2])
        #print('Max:',np.max(img[:,:,0]), np.max(img[:,:,1]), np.max(img[:,:,2]))
        #print('Min:',np.min(img[:,:,0]), np.min(img[:,:,1]), np.min(img[:,:,2]))
    #Process Grayscale Images
    elif colorScheme=='gray':
        #Read image
        img = cv2.cvtColor(cv2.imread(str(imgPath)), cv2.COLOR_BGR2GRAY)
        width, height = img.shape
        #Crop to required size
        w_crop = int(np.floor((width-imgSize[0])/2))
        h_crop = int(np.floor((height-imgSize[1])/2))
        img = img[w_crop:w_crop+imgSize[0], h_crop:h_crop+imgSize[1]]
        #Convert to Float 32
        img = np.array(img, dtype=np.float32)
        #Rescale
        img[:,:] = img[:,:]/255.0
        #Subtract mean
        if subtractMean:
            img[:,:] = img[:,:] - np.mean(img[:,:])
        #Divide by standard deviation
        if divideStdDev:
            img[:,:] = img[:,:] / np.std(img[:,:])
        #Expand dim to fit Tensorflow shape requirements
        img = np.expand_dims(img, axis=2)#Grayscale
    else:
        raise('Invalid color scheme')
    return img

def evaluateModel(modelPath, imgPaths):
    model = load_model(str(modelPath))
    #Metrics
    falseNegative = 0
    trueNegative = 0
    falsePositive = 0
    truePositive = 0
    y_true = []
    y_pred = []
    for imgPath in imgPaths:
        img = processImg(imgPath, subtractMean=True, divideStdDev=False, colorScheme='rgb', imgSize=(250,250))
        prediction = model.predict(np.array([img],dtype=np.float32), verbose=1)
        #Model Predict ALL...
        if prediction[0][0] > prediction[0][1]:
            if 'all.bmp' in imgPath.name:#... and cell type is ALL => True Positive
                truePositive += 1
            if 'hem.bmp' in imgPath.name:#... and cell type is HEM => False Positive
                falsePositive += 1
        #Model Predict HEM...
        elif prediction[0][1] > prediction[0][0]:
            if 'all.bmp' in imgPath.name:#... and cell type is ALL => False Negative
                falseNegative += 1
            if 'hem.bmp' in imgPath.name:#... and cell type is HEM => True Negative
                trueNegative += 1
    #Epsilon avoid division by zero
    epsilon = 1e-9
    #sensitivity/recall =  measures the proportion of actual positives that are correctly identified as such
    sensitivity = truePositive/(truePositive+falseNegative+epsilon)
    #specificity = measures the proportion of actual negatives that are correctly identified as such
    specificity = trueNegative/(trueNegative+falsePositive+epsilon)
    #accuracy = measures the systematic error
    accuracy = (truePositive + trueNegative)/(truePositive + trueNegative + falsePositive + falseNegative+epsilon)
    #precision = description of random errors, a measure of statistical variability.
    precision = truePositive/(truePositive + falsePositive+epsilon)

    sensitivity = np.round((sensitivity*100.0), decimals=2)
    specificity = np.round((specificity*100.0), decimals=2)
    accuracy = np.round((accuracy*100.0), decimals=2)
    precision = np.round((precision*100.0), decimals=2)
    F1_score = np.round(2*((precision*sensitivity)/(precision+sensitivity)), decimals=2)

    print('###################')
    print(f'Sensitivity: {sensitivity}%')
    print(f'Specificity: {specificity}%')
    print(f'Accuracy: {accuracy}%')
    print(f'Precision: {precision}%')
    print(f'F1-Score: {F1_score}%')
    print('###################')
    print(f'True Positive: {truePositive}')
    print(f'True Negative: {trueNegative}')
    print(f'False Positive: {falsePositive}')
    print(f'False Negative: {falseNegative}')
    print('###################')
    
    K.clear_session()
    return sensitivity, specificity, accuracy, precision, F1_score


if __name__ == '__main__':
    augm_patLvDiv_valid = Path('../../data/augm_patLvDiv_valid')
    augm_rndDiv_valid = Path('../../data/augm_rndDiv_valid')

    augmPatLvDiv_validImgs = []
    for imgFile in os.listdir(augm_patLvDiv_valid):
        augmPatLvDiv_validImgs.append(augm_patLvDiv_valid / imgFile)
    
    augmRndDiv_validImgs = []
    for imgFile in os.listdir(augm_rndDiv_valid):
        augmRndDiv_validImgs.append(augm_rndDiv_valid / imgFile)

    np.random.shuffle(augmPatLvDiv_validImgs)
    np.random.shuffle(augmRndDiv_validImgs)

    ################################################################################################
    results = pd.DataFrame()
    ################################################################################################
    sens, spec, acc, prec, F1sco = evaluateModel(xception_AugmPatDiv, augmPatLvDiv_validImgs)
    collectedData = {}
    collectedData['ConvNet'] = 'xception'
    collectedData['test-ID']= 'PatDiv'
    collectedData['sensitivity'] = sens
    collectedData['specificity'] = spec
    collectedData['accuracy'] = acc
    collectedData['precision'] = prec
    collectedData['F1_score'] = F1sco
    results = results.append(collectedData, ignore_index=True)
    results.to_csv(Path('results/ConvNet_ValidPerformance.csv'))
    ################################################################################################
    sens, spec, acc, prec, F1sco = evaluateModel(xception_AugmPatDiv_nvidia1070, augmPatLvDiv_validImgs)
    collectedData = {}
    collectedData['ConvNet'] = 'xception-nvidia1070'
    collectedData['test-ID']= 'PatDiv'
    collectedData['sensitivity'] = sens
    collectedData['specificity'] = spec
    collectedData['accuracy'] = acc
    collectedData['precision'] = prec
    collectedData['F1_score'] = F1sco
    results = results.append(collectedData, ignore_index=True)
    results.to_csv(Path('results/ConvNet_ValidPerformance.csv'))
    ################################################################################################
    sens, spec, acc, prec, F1sco = evaluateModel(xception_AugmRndDiv, augmRndDiv_validImgs)
    collectedData = {}
    collectedData['ConvNet'] = 'xception'
    collectedData['test-ID']= 'RndDiv'
    collectedData['sensitivity'] = sens
    collectedData['specificity'] = spec
    collectedData['accuracy'] = acc
    collectedData['precision'] = prec
    collectedData['F1_score'] = F1sco
    results = results.append(collectedData, ignore_index=True)
    results.to_csv(Path('results/ConvNet_ValidPerformance.csv'))
    ################################################################################################
    sens, spec, acc, prec, F1sco = evaluateModel(xception_AugmRndDiv_nvidia1070, augmRndDiv_validImgs)
    collectedData = {}
    collectedData['ConvNet'] = 'xception-nvidia1070'
    collectedData['test-ID']= 'RndDiv'
    collectedData['sensitivity'] = sens
    collectedData['specificity'] = spec
    collectedData['accuracy'] = acc
    collectedData['precision'] = prec
    collectedData['F1_score'] = F1sco
    results = results.append(collectedData, ignore_index=True)
    results.to_csv(Path('results/ConvNet_ValidPerformance.csv'))
    ################################################################################################
    #**********************************************************************************************#
    ################################################################################################

    sens, spec, acc, prec, F1sco = evaluateModel(vgg16_AugmPatDiv, augmPatLvDiv_validImgs)
    collectedData = {}
    collectedData['ConvNet'] = 'vgg16'
    collectedData['test-ID']= 'PatDiv'
    collectedData['sensitivity'] = sens
    collectedData['specificity'] = spec
    collectedData['accuracy'] = acc
    collectedData['precision'] = prec
    collectedData['F1_score'] = F1sco
    results = results.append(collectedData, ignore_index=True)
    results.to_csv(Path('results/ConvNet_ValidPerformance.csv'))
    ################################################################################################
    sens, spec, acc, prec, F1sco = evaluateModel(vgg16_AugmPatDiv_nvidia1070, augmPatLvDiv_validImgs)
    collectedData = {}
    collectedData['ConvNet'] = 'vgg16-nvidia1070'
    collectedData['test-ID']= 'PatDiv'
    collectedData['sensitivity'] = sens
    collectedData['specificity'] = spec
    collectedData['accuracy'] = acc
    collectedData['precision'] = prec
    collectedData['F1_score'] = F1sco
    results = results.append(collectedData, ignore_index=True)
    results.to_csv(Path('results/ConvNet_ValidPerformance.csv'))
    ################################################################################################
    sens, spec, acc, prec, F1sco = evaluateModel(vgg16_AugmRndDiv, augmRndDiv_validImgs)
    collectedData = {}
    collectedData['ConvNet'] = 'vgg16'
    collectedData['test-ID']= 'RndDiv'
    collectedData['sensitivity'] = sens
    collectedData['specificity'] = spec
    collectedData['accuracy'] = acc
    collectedData['precision'] = prec
    collectedData['F1_score'] = F1sco
    results = results.append(collectedData, ignore_index=True)
    results.to_csv(Path('results/ConvNet_ValidPerformance.csv'))
    ################################################################################################
    sens, spec, acc, prec, F1sco = evaluateModel(vgg16_AugmRndDiv_nvidia1070, augmRndDiv_validImgs)
    collectedData = {}
    collectedData['ConvNet'] = 'vgg16-nvidia1070'
    collectedData['test-ID']= 'RndDiv'
    collectedData['sensitivity'] = sens
    collectedData['specificity'] = spec
    collectedData['accuracy'] = acc
    collectedData['precision'] = prec
    collectedData['F1_score'] = F1sco
    results = results.append(collectedData, ignore_index=True)
    results.to_csv(Path('results/ConvNet_ValidPerformance.csv'))

    ################################################################################################
    #**********************************************************************************************#
    ################################################################################################

    sens, spec, acc, prec, F1sco = evaluateModel(vgg19_AugmPatDiv, augmPatLvDiv_validImgs)
    collectedData = {}
    collectedData['ConvNet'] = 'vgg19'
    collectedData['test-ID']= 'PatDiv'
    collectedData['sensitivity'] = sens
    collectedData['specificity'] = spec
    collectedData['accuracy'] = acc
    collectedData['precision'] = prec
    collectedData['F1_score'] = F1sco
    results = results.append(collectedData, ignore_index=True)
    results.to_csv(Path('results/ConvNet_ValidPerformance.csv'))
    ################################################################################################
    sens, spec, acc, prec, F1sco = evaluateModel(vgg19_AugmPatDiv_nvidia1070, augmPatLvDiv_validImgs)
    collectedData = {}
    collectedData['ConvNet'] = 'vgg19-nvidia1070'
    collectedData['test-ID']= 'PatDiv'
    collectedData['sensitivity'] = sens
    collectedData['specificity'] = spec
    collectedData['accuracy'] = acc
    collectedData['precision'] = prec
    collectedData['F1_score'] = F1sco
    results = results.append(collectedData, ignore_index=True)
    results.to_csv(Path('results/ConvNet_ValidPerformance.csv'))
    ################################################################################################
    sens, spec, acc, prec, F1sco = evaluateModel(vgg19_AugmRndDiv, augmRndDiv_validImgs)
    collectedData = {}
    collectedData['ConvNet'] = 'vgg19'
    collectedData['test-ID']= 'RndDiv'
    collectedData['sensitivity'] = sens
    collectedData['specificity'] = spec
    collectedData['accuracy'] = acc
    collectedData['precision'] = prec
    collectedData['F1_score'] = F1sco
    results = results.append(collectedData, ignore_index=True)
    results.to_csv(Path('results/ConvNet_ValidPerformance.csv'))
    ################################################################################################
    sens, spec, acc, prec, F1sco = evaluateModel(vgg19_AugmRndDiv_nvidia1070, augmRndDiv_validImgs)
    collectedData = {}
    collectedData['ConvNet'] = 'vgg19-nvidia1070'
    collectedData['test-ID']= 'RndDiv'
    collectedData['sensitivity'] = sens
    collectedData['specificity'] = spec
    collectedData['accuracy'] = acc
    collectedData['precision'] = prec
    collectedData['F1_score'] = F1sco
    results = results.append(collectedData, ignore_index=True)
    results.to_csv(Path('results/ConvNet_ValidPerformance.csv'))

    ################################################################################################
    #**********************************************************************************************#
    ################################################################################################

    sens, spec, acc, prec, F1sco = evaluateModel(resNet50_AugmPatDiv, augmPatLvDiv_validImgs)
    collectedData = {}
    collectedData['ConvNet'] = 'resNet50'
    collectedData['test-ID']= 'PatDiv'
    collectedData['sensitivity'] = sens
    collectedData['specificity'] = spec
    collectedData['accuracy'] = acc
    collectedData['precision'] = prec
    collectedData['F1_score'] = F1sco
    results = results.append(collectedData, ignore_index=True)
    results.to_csv(Path('results/ConvNet_ValidPerformance.csv'))
    ################################################################################################
    sens, spec, acc, prec, F1sco = evaluateModel(resNet50_AugmRndDiv, augmRndDiv_validImgs)
    collectedData = {}
    collectedData['ConvNet'] = 'resNet50'
    collectedData['test-ID']= 'RndDiv'
    collectedData['sensitivity'] = sens
    collectedData['specificity'] = spec
    collectedData['accuracy'] = acc
    collectedData['precision'] = prec
    collectedData['F1_score'] = F1sco
    results = results.append(collectedData, ignore_index=True)
    results.to_csv(Path('results/ConvNet_ValidPerformance.csv'))

    ################################################################################################
    #**********************************************************************************************#
    ################################################################################################

    sens, spec, acc, prec, F1sco = evaluateModel(inceptionV3_AugmPatDiv, augmPatLvDiv_validImgs)
    collectedData = {}
    collectedData['ConvNet'] = 'inceptionV3'
    collectedData['test-ID']= 'PatDiv'
    collectedData['sensitivity'] = sens
    collectedData['specificity'] = spec
    collectedData['accuracy'] = acc
    collectedData['precision'] = prec
    collectedData['F1_score'] = F1sco
    results = results.append(collectedData, ignore_index=True)
    results.to_csv(Path('results/ConvNet_ValidPerformance.csv'))
    ################################################################################################
    sens, spec, acc, prec, F1sco = evaluateModel(inceptionV3_AugmRndDiv, augmRndDiv_validImgs)
    collectedData = {}
    collectedData['ConvNet'] = 'inceptionV3'
    collectedData['test-ID']= 'RndDiv'
    collectedData['sensitivity'] = sens
    collectedData['specificity'] = spec
    collectedData['accuracy'] = acc
    collectedData['precision'] = prec
    collectedData['F1_score'] = F1sco
    results = results.append(collectedData, ignore_index=True)


    print(results.head())
    print(results.tail())


    results.to_csv(Path('results/ConvNet_ValidPerformance.csv'))

    print(f"\nEnd Script!\n{'#'*50}")
