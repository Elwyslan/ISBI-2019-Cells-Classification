import os
import cv2
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Model, load_model
from A_executeOnValidation import evaluateModel
import pandas as pd

ALL_LABEL = [[1.0, 0.0]]
HEM_LABEL = [[0.0, 1.0]]

patLvDiv_test = Path('../../data/patLvDiv_test')
rndDiv_test = Path('../../data/rndDiv_test')

xception_AugmPatDiv = Path('Best Models/xception_AugmPatDiv.h5')
xception_AugmRndDiv = Path('Best Models/xception_augmRndDiv.h5')

vgg16_AugmPatDiv = Path('Best Models/VGG16_AugmPatDiv.h5')
vgg16_AugmRndDiv = Path('Best Models/VGG16_AugmRndDiv.h5')

vgg19_AugmPatDiv = Path('Best Models/VGG19_augmPatLvDiv.h5')
vgg19_AugmRndDiv = Path('Best Models/VGG19_augmRndDiv.h5')

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

def checkBalance(imgList):
    countALL = countHEM = 0
    for imgPath in imgList:
        if 'all.bmp' in imgPath.name:
            countALL+=1
        elif 'hem.bmp' in imgPath.name:
            countHEM+=1
    print(f'Counter: {countALL}-ALL, {countHEM}-HEM')

def balanceList(listImgsPath):
    listImgsPath = listImgsPath.copy()
    countALL = countHEM = 0
    for imgPath in listImgsPath:
        if 'all.bmp' in imgPath.name:
            countALL+=1
        elif 'hem.bmp' in imgPath.name:
            countHEM+=1
    #Drop HEM
    if countHEM>countALL:
        dropHEM = abs(countHEM-countALL)
        HEMidx = []
        for i in range(len(listImgsPath)):
            if 'hem.bmp' in listImgsPath[i].name:
                HEMidx.append(i)
        np.random.shuffle(HEMidx)
        np.random.shuffle(HEMidx)
        removeHEM = []
        for i in range(dropHEM):
            removeHEM.append(listImgsPath[HEMidx[i]])
        for item in removeHEM:
            listImgsPath.remove(item)
        return listImgsPath
    #Drop ALL
    if countALL>countHEM:
        dropALL = abs(countHEM-countALL)
        ALLidx = []
        for i in range(len(listImgsPath)):
            if 'all.bmp' in listImgsPath[i].name:
                ALLidx.append(i)
        np.random.shuffle(ALLidx)
        np.random.shuffle(ALLidx)
        removeALL = []
        for i in range(dropALL):
            removeALL.append(listImgsPath[ALLidx[i]])
        for item in removeALL:
            listImgsPath.remove(item)
        return listImgsPath
    #No drop
    if countALL==countHEM:
        return listImgsPath



if __name__ == '__main__':
    patLvTestImgs = []
    for imgFile in os.listdir(patLvDiv_test):
        patLvTestImgs.append(patLvDiv_test / imgFile)
    rndDivTestImgs = []
    for imgFile in os.listdir(rndDiv_test):
        rndDivTestImgs.append(rndDiv_test / imgFile)
    np.random.shuffle(patLvTestImgs)
    np.random.shuffle(rndDivTestImgs)
    #pImg = processImg(patLvTestImgs[0], subtractMean=True, divideStdDev=True, colorScheme='rgb', imgSize=(250,250))
    #plt.subplot(211)
    #plt.hist(pImg[:,:,0])
    #print(f'Mean:{np.mean(pImg[:,:,0])}, Std:{np.std(pImg[:,:,0])}')
    #pImg = processImg(patLvTestImgs[0], subtractMean=True, divideStdDev=False, colorScheme='rgb', imgSize=(250,250))
    #plt.subplot(212)
    #plt.hist(pImg[:,:,0])
    #print(f'Mean:{np.mean(pImg[:,:,0])}, Std:{np.std(pImg[:,:,0])}')
    #plt.show()
    #quit()
    
    checkBalance(patLvTestImgs)
    checkBalance(rndDivTestImgs)
    patLvTestImgs = balanceList(patLvTestImgs)
    rndDivTestImgs = balanceList(rndDivTestImgs)
    checkBalance(patLvTestImgs)
    checkBalance(rndDivTestImgs)

    ################################################################################################
    results = pd.DataFrame()
    ################################################################################################
    sens, spec, acc, prec, F1sco = evaluateModel(xception_AugmPatDiv, patLvTestImgs)
    collectedData = {}
    collectedData['ConvNet'] = 'xception'
    collectedData['test-ID']= 'PatDiv'
    collectedData['sensitivity'] = sens
    collectedData['specificity'] = spec
    collectedData['accuracy'] = acc
    collectedData['precision'] = prec
    collectedData['F1_score'] = F1sco
    results = results.append(collectedData, ignore_index=True)
    ################################################################################################
    sens, spec, acc, prec, F1sco = evaluateModel(xception_AugmRndDiv, rndDivTestImgs)
    collectedData = {}
    collectedData['ConvNet'] = 'xception'
    collectedData['test-ID']= 'RndDiv'
    collectedData['sensitivity'] = sens
    collectedData['specificity'] = spec
    collectedData['accuracy'] = acc
    collectedData['precision'] = prec
    collectedData['F1_score'] = F1sco
    results = results.append(collectedData, ignore_index=True)
    ################################################################################################
    #**********************************************************************************************#
    ################################################################################################
    sens, spec, acc, prec, F1sco = evaluateModel(vgg16_AugmPatDiv, patLvTestImgs)
    collectedData = {}
    collectedData['ConvNet'] = 'vgg16'
    collectedData['test-ID']= 'PatDiv'
    collectedData['sensitivity'] = sens
    collectedData['specificity'] = spec
    collectedData['accuracy'] = acc
    collectedData['precision'] = prec
    collectedData['F1_score'] = F1sco
    results = results.append(collectedData, ignore_index=True)
    ################################################################################################
    sens, spec, acc, prec, F1sco = evaluateModel(vgg16_AugmRndDiv, rndDivTestImgs)
    collectedData = {}
    collectedData['ConvNet'] = 'vgg16'
    collectedData['test-ID']= 'RndDiv'
    collectedData['sensitivity'] = sens
    collectedData['specificity'] = spec
    collectedData['accuracy'] = acc
    collectedData['precision'] = prec
    collectedData['F1_score'] = F1sco
    results = results.append(collectedData, ignore_index=True)
    ################################################################################################
    #**********************************************************************************************#
    ################################################################################################
    sens, spec, acc, prec, F1sco = evaluateModel(vgg19_AugmPatDiv, patLvTestImgs)
    collectedData = {}
    collectedData['ConvNet'] = 'vgg19'
    collectedData['test-ID']= 'PatDiv'
    collectedData['sensitivity'] = sens
    collectedData['specificity'] = spec
    collectedData['accuracy'] = acc
    collectedData['precision'] = prec
    collectedData['F1_score'] = F1sco
    results = results.append(collectedData, ignore_index=True)
    ################################################################################################
    sens, spec, acc, prec, F1sco = evaluateModel(vgg19_AugmRndDiv, rndDivTestImgs)
    collectedData = {}
    collectedData['ConvNet'] = 'vgg19'
    collectedData['test-ID']= 'RndDiv'
    collectedData['sensitivity'] = sens
    collectedData['specificity'] = spec
    collectedData['accuracy'] = acc
    collectedData['precision'] = prec
    collectedData['F1_score'] = F1sco
    results = results.append(collectedData, ignore_index=True)

    print(results.head())
    print(results.tail())

    results.to_csv(Path('results/ConvNet_TestPerformance.csv'))

    print(f"\nEnd Script!\n{'#'*50}")
