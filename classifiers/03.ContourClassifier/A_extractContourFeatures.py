import os
import multiprocessing
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import radiomics
radiomics.setVerbosity(60)#Quiet
import SimpleITK as sitk
from matplotlib import pyplot as plt
from scipy.spatial import distance
from scipy.signal import resample

augm_patLvDiv_train = Path('../../data/augm_patLvDiv_train/')
augm_patLvDiv_valid = Path('../../data/augm_patLvDiv_valid')

augm_rndDiv_train = Path('../../data/augm_rndDiv_train')
augm_rndDiv_valid = Path('../../data/augm_rndDiv_valid')

patLvDiv_train = Path('../../data/patLvDiv_train')
patLvDiv_valid = Path('../../data/patLvDiv_valid')
patLvDiv_test = Path('../../data/patLvDiv_test')

rndDiv_train = Path('../../data/rndDiv_train')
rndDiv_valid = Path('../../data/rndDiv_valid')
rndDiv_test = Path('../../data/rndDiv_test')

def getQtdSamples():
    f0 = os.listdir(augm_patLvDiv_train)
    f1 = os.listdir(augm_patLvDiv_valid)
    f2 = os.listdir(augm_rndDiv_train)
    f3 = os.listdir(augm_rndDiv_valid)

    def getSignatureQtdSamples(path):
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2GRAY)
        _,thresh = cv2.threshold(img, 1, 1, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, np.ones((2,2),np.uint8),iterations = 1)
        _, contours, _ = cv2.findContours(thresh, 1, 2)
        contour = max(contours, key=cv2.contourArea)
        return len(contour)

    lenghts = []
    for n, imgFile in enumerate(f0):
        imgPath = augm_patLvDiv_train / imgFile
        lenghts.append(getSignatureQtdSamples(imgPath))
        print(n)
    for n, imgFile in enumerate(f1):
        imgPath = augm_patLvDiv_valid / imgFile
        lenghts.append(getSignatureQtdSamples(imgPath))
        print(n)
    for n, imgFile in enumerate(f2):
        imgPath = augm_rndDiv_train / imgFile
        lenghts.append(getSignatureQtdSamples(imgPath))
        print(n)
    for n, imgFile in enumerate(f3):
        imgPath = augm_rndDiv_valid / imgFile
        lenghts.append(getSignatureQtdSamples(imgPath))
        print(n)
    
    print(f'Mean lenght = {np.mean(lenghts)}, std. lenght = {np.std(lenghts)}')

    return np.mean(lenghts), np.std(lenghts)



def getContourSignature(grayScaleImage, sizeVector=320):
    _,thresh = cv2.threshold(grayScaleImage, 1, 255, cv2.THRESH_BINARY)
    thresh = cv2.erode(thresh, np.ones((2,2),np.uint8),iterations = 1)
    _, contours, _ = cv2.findContours(thresh, 1, 2)
    contour = max(contours, key=cv2.contourArea)
    m = cv2.moments(thresh)
    centroid = (int(m['m10']/m['m00']), int(m['m01']/m['m00']))
    signature = []
    for points in contour:
        curvePoint = points[0][0], points[0][1]
        signature.append(distance.euclidean(centroid, curvePoint))
    if (len(signature)>sizeVector) or (len(signature)<sizeVector):
        signature = resample(signature, sizeVector)
    signature = list(map(abs, np.fft.fft(signature)))
    signature = signature[0:len(signature)//2]
    return signature

def getContourFeatures(imagePath):
    features = {}
    if 'all.bmp' in imagePath.name:
        features['cellType(ALL=1, HEM=-1)'] = 1
    elif 'hem.bmp' in imagePath.name:
        features['cellType(ALL=1, HEM=-1)'] = -1
    else:
        return features
    img = cv2.cvtColor(cv2.imread(str(imagePath)), cv2.COLOR_BGR2GRAY)
    signature = getContourSignature(img)
    for i in range(len(signature)):
        features[f'pt{str(i).zfill(3)}'] = signature[i]
    return features

"""
###############################################################################################
###############################################################################################
###############################################################################################
"""
def createAugmPatLvDivDataframe():
    TRAIN_IMGS = os.listdir(augm_patLvDiv_train)
    VALID_IMGS = os.listdir(augm_patLvDiv_valid)
    #Create Train Dataframe
    def createTrainDataframe():
        train_df = pd.DataFrame()
        np.random.shuffle(TRAIN_IMGS)
        for n, imgFile in enumerate(TRAIN_IMGS):
            imgPath = augm_patLvDiv_train / imgFile
            featuresDict = getContourFeatures(imgPath)
            train_df = train_df.append(featuresDict, ignore_index=True)
            print(f'{n}-Train')
        cols = train_df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        train_df.to_csv(f'data/AugmPatLvDiv_TRAIN-ContourData_{train_df.shape[1]-1}-Features_{train_df.shape[0]}-images.csv')

    #Process for Validation dataset
    def createValidDataframe():
        valid_df = pd.DataFrame()
        np.random.shuffle(VALID_IMGS)
        for n, imgFile in enumerate(VALID_IMGS):
            imgPath = augm_patLvDiv_valid / imgFile
            featuresDict = getContourFeatures(imgPath)
            valid_df = valid_df.append(featuresDict, ignore_index=True)
            print(f'{n}-Validation')
        cols = valid_df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        valid_df = valid_df[cols]
        valid_df.to_csv(f'data/AugmPatLvDiv_VALIDATION-ContourData_{valid_df.shape[1]-1}-Features_{valid_df.shape[0]}-images.csv')
    
    p0 = multiprocessing.Process(name='train_AugmPatLvDiv', target=createTrainDataframe)
    p1 = multiprocessing.Process(name='valid_AugmPatLvDiv',target=createValidDataframe)
    p0.start()
    p1.start()
    p0.join()
    p1.join()

"""
###############################################################################################
###############################################################################################
###############################################################################################
"""

def createAugmRndDivDataframe():
    TRAIN_IMGS = os.listdir(augm_rndDiv_train)
    VALID_IMGS = os.listdir(augm_rndDiv_valid)
    #Create Train Dataframe
    def createTrainDataframe():
        train_df = pd.DataFrame()
        np.random.shuffle(TRAIN_IMGS)
        for n, imgFile in enumerate(TRAIN_IMGS):
            imgPath = augm_rndDiv_train / imgFile
            featuresDict = getContourFeatures(imgPath)
            train_df = train_df.append(featuresDict, ignore_index=True)
            print(f'{n}-Train')
        cols = train_df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        train_df = train_df[cols]
        train_df.to_csv(f'data/AugmRndDiv_TRAIN-ContourData_{train_df.shape[1]-1}-Features_{train_df.shape[0]}-images.csv')

    #Process for Validation dataset
    def createValidDataframe():
        valid_df = pd.DataFrame()
        np.random.shuffle(VALID_IMGS)
        for n, imgFile in enumerate(VALID_IMGS):
            imgPath = augm_rndDiv_valid/ imgFile
            featuresDict = getContourFeatures(imgPath)
            valid_df = valid_df.append(featuresDict, ignore_index=True)
            print(f'{n}-Validation')
        cols = valid_df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        valid_df = valid_df[cols]
        valid_df.to_csv(f'data/AugmRndDiv_VALIDATION-ContourData_{valid_df.shape[1]-1}-Features_{valid_df.shape[0]}-images.csv')
    
    p0 = multiprocessing.Process(name='train_AugmRndDiv', target=createTrainDataframe)
    p1 = multiprocessing.Process(name='valid_AugmRndDiv',target=createValidDataframe)
    p0.start()
    p1.start()
    p0.join()
    p1.join()
    
"""
###############################################################################################
###############################################################################################
###############################################################################################
"""

def createPatLvDivDataframe():
    TRAIN_IMGS = os.listdir(patLvDiv_train)
    VALID_IMGS = os.listdir(patLvDiv_valid)
    TEST_IMGS = os.listdir(patLvDiv_test)
    #Create Train Dataframe
    def createTrainDataframe():
        train_df = pd.DataFrame()
        np.random.shuffle(TRAIN_IMGS)
        for n, imgFile in enumerate(TRAIN_IMGS):
            imgPath = patLvDiv_train / imgFile
            featuresDict = getContourFeatures(imgPath)
            train_df = train_df.append(featuresDict, ignore_index=True)
            print(f'{n}-Train')
        cols = train_df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        train_df = train_df[cols]
        train_df.to_csv(f'data/PatLvDiv_TRAIN-ContourData_{train_df.shape[1]-1}-Features_{train_df.shape[0]}-images.csv')

    #Process for Validation dataset
    def createValidDataframe():
        valid_df = pd.DataFrame()
        np.random.shuffle(VALID_IMGS)
        for n, imgFile in enumerate(VALID_IMGS):
            featuresDict = {}
            imgPath = patLvDiv_valid / imgFile
            featuresDict = getContourFeatures(imgPath)
            valid_df = valid_df.append(featuresDict, ignore_index=True)
            print(f'{n}-Validation')
        cols = valid_df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        valid_df = valid_df[cols]
        valid_df.to_csv(f'data/PatLvDiv_VALIDATION-ContourData_{valid_df.shape[1]-1}-Features_{valid_df.shape[0]}-images.csv')

    #Process for Test dataset
    def createTestDataframe():
        test_df = pd.DataFrame()
        np.random.shuffle(TEST_IMGS)
        for n, imgFile in enumerate(TEST_IMGS):
            featuresDict = {}
            imgPath = patLvDiv_test / imgFile
            featuresDict = getContourFeatures(imgPath)
            test_df = test_df.append(featuresDict, ignore_index=True)
            print(f'{n}-Test')
        cols = test_df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        test_df = test_df[cols]
        test_df.to_csv(f'data/PatLvDiv_TEST-ContourData_{test_df.shape[1]-1}-Features_{test_df.shape[0]}-images.csv')
    
    p0 = multiprocessing.Process(name='train_PatLvDiv', target=createTrainDataframe)
    p1 = multiprocessing.Process(name='valid_PatLvDiv',target=createValidDataframe)
    p2 = multiprocessing.Process(name='test_PatLvDiv',target=createTestDataframe)
    p0.start()
    p1.start()
    p2.start()
    p0.join()
    p1.join()
    p2.join()


"""
###############################################################################################
###############################################################################################
###############################################################################################
"""

def createRndDivDataframe():
    TRAIN_IMGS = os.listdir(rndDiv_train)
    VALID_IMGS = os.listdir(rndDiv_valid)
    TEST_IMGS = os.listdir(rndDiv_test)
    #Create Train Dataframe
    def createTrainDataframe():
        train_df = pd.DataFrame()
        np.random.shuffle(TRAIN_IMGS)
        for n, imgFile in enumerate(TRAIN_IMGS):
            featuresDict = {}
            imgPath = rndDiv_train / imgFile
            featuresDict = getContourFeatures(imgPath)
            train_df = train_df.append(featuresDict, ignore_index=True)
            print(f'{n}-Train')
        cols = train_df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        train_df = train_df[cols]
        train_df.to_csv(f'data/rndDiv_TRAIN-ContourData_{train_df.shape[1]-1}-Features_{train_df.shape[0]}-images.csv')

    #Process for Validation dataset
    def createValidDataframe():
        valid_df = pd.DataFrame()
        np.random.shuffle(VALID_IMGS)
        for n, imgFile in enumerate(VALID_IMGS):
            featuresDict = {}
            imgPath = rndDiv_valid / imgFile
            featuresDict = getContourFeatures(imgPath)
            valid_df = valid_df.append(featuresDict, ignore_index=True)
            print(f'{n}-Validation')
        cols = valid_df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        valid_df = valid_df[cols]
        valid_df.to_csv(f'data/rndDiv_VALIDATION-ContourData_{valid_df.shape[1]-1}-Features_{valid_df.shape[0]}-images.csv')

    #Process for Test dataset
    def createTestDataframe():
        test_df = pd.DataFrame()
        np.random.shuffle(TEST_IMGS)
        for n, imgFile in enumerate(TEST_IMGS):
            featuresDict = {}
            imgPath = rndDiv_test / imgFile
            featuresDict = getContourFeatures(imgPath)
            test_df = test_df.append(featuresDict, ignore_index=True)
            print(f'{n}-Test')
        cols = test_df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        test_df = test_df[cols]
        test_df.to_csv(f'data/rndDiv_TEST-ContourData_{test_df.shape[1]-1}-Features_{test_df.shape[0]}-images.csv')
    
    p0 = multiprocessing.Process(name='train_RndDiv', target=createTrainDataframe)
    p1 = multiprocessing.Process(name='valid_RndDiv',target=createValidDataframe)
    p2 = multiprocessing.Process(name='test_RndDiv',target=createTestDataframe)
    p0.start()
    p1.start()
    p2.start()
    p0.join()
    p1.join()
    p2.join()

"""
###############################################################################################
###############################################################################################
###############################################################################################
"""

if __name__ == '__main__':
    s0 = Path('../../data/augm_rndDiv_train/AugmentedImg_1051_UID_H24_33_11_hem.bmp')
    assert len(getContourFeatures(s0))==161, 'Erro, check feature extraction'
    #Spawn Process
    p0 = multiprocessing.Process(name='AugmPatLvDiv', target=createAugmPatLvDivDataframe)
    p1 = multiprocessing.Process(name='AugmRndDiv',target=createAugmRndDivDataframe)
    p2 = multiprocessing.Process(name='PatLvDiv', target=createPatLvDivDataframe)
    p3 = multiprocessing.Process(name='RndDiv',target=createRndDivDataframe)

    p0.start()
    p1.start()
    p2.start()
    p3.start()

    p0.join()
    p1.join()
    p2.join()
    p3.join()
    
    print(f"\nEnd Script!\n{'#'*50}")
