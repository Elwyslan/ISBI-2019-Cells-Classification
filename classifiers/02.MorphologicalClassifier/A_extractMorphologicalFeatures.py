import os
import multiprocessing
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import SimpleITK as sitk
from matplotlib import pyplot as plt

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

def getNucleusArea(grayScaleImage):
    _, thresh = cv2.threshold(grayScaleImage, 1, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    return cv2.contourArea(contour)

def getNucleusPerimeter(grayScaleImage):
    _, thresh = cv2.threshold(grayScaleImage, 1, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    return cv2.arcLength(contour, True)

def getConvexArea(grayScaleImage):
    _, thresh = cv2.threshold(grayScaleImage, 1, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour, returnPoints=True)
    return cv2.contourArea(hull)

def getConvexPerimeter(grayScaleImage):
    _,thresh = cv2.threshold(grayScaleImage, 1, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour, returnPoints=True)
    return cv2.arcLength(hull, True)

#DOI 10.1002/jemt.22718
def getEquivalentDiameter(grayScaleImage):
    return np.sqrt(4.0 * (getNucleusArea(grayScaleImage)/np.pi))

#DOI 10.1016/j.artmed.2014.09.002 **** DOI 10.1002/jemt.22718
def getCompactness(grayScaleImage):
    return (4.0 * np.pi * getNucleusArea(grayScaleImage)) / np.power(getNucleusPerimeter(grayScaleImage), 2)

#DOI 10.1016/j.artmed.2014.09.002 **** DOI 10.1002/jemt.22718
def getConvexity(grayScaleImage):
    return getConvexPerimeter(grayScaleImage) / getNucleusPerimeter(grayScaleImage)

#DOI 10.1016/j.artmed.2014.09.002 **** DOI 10.1002/jemt.22718
def getSolidity(grayScaleImage):
    return getNucleusArea(grayScaleImage) / getConvexArea(grayScaleImage)

#DOI 10.1016/j.artmed.2014.09.002 **** DOI 10.1002/jemt.22718
def getNucleosEccentricity(grayScaleImage):
    _, thresh = cv2.threshold(grayScaleImage, 1, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    (x, y), (minorAxis, majorAxis), _ = cv2.fitEllipse(contour)
    a = majorAxis/2
    b = minorAxis/2
    c = np.sqrt(a**2 - b**2)
    eccentricity = c/a
    aspectRatio = minorAxis/majorAxis
    return eccentricity, aspectRatio, majorAxis, minorAxis

#DOI 10.1016/j.artmed.2014.09.002
def getNucleusElongation(grayScaleImage):
    _, _, majorAxis, minorAxis = getNucleosEccentricity(grayScaleImage)
    return 1.0 - (minorAxis/majorAxis)

#DOI 10.1016/j.artmed.2014.09.002
def getRectangularity(grayScaleImage):
    area = getNucleusArea(grayScaleImage)
    _, _, majorAxis, minorAxis = getNucleosEccentricity(grayScaleImage)
    return area/(majorAxis*minorAxis)

def getMorphologicalFeatures(imagePath):
    features = {}
    if 'all.bmp' in imagePath.name:
        features['cellType(ALL=1, HEM=-1)'] = 1
    elif 'hem.bmp' in imagePath.name:
        features['cellType(ALL=1, HEM=-1)'] = -1
    else:
        return {}

    srcImg = cv2.imread(str(imagePath))
    grayImg = cv2.cvtColor(srcImg, cv2.COLOR_BGR2GRAY)
    h, s, v = cv2.split(cv2.cvtColor(srcImg, cv2.COLOR_BGR2HSV))
    b, g, r = cv2.split(srcImg)
    #Color Features
    features['morphFeats_meanHue'] = np.mean(h)
    features['morphFeats_meanSaturation'] = np.mean(s)
    features['morphFeats_meanValue'] = np.mean(v)
    features['morphFeats_meanRed'] = np.mean(r)
    features['morphFeats_meanGreen'] = np.mean(g)
    features['morphFeats_meanBlue'] = np.mean(b)
    #Shape Features
    features['morphFeats_nucleoArea'] = getNucleusArea(grayImg)
    features['morphFeats_nucleoPerimeter'] = getNucleusPerimeter(grayImg)
    features['morphFeats_convexArea'] = getConvexArea(grayImg)
    features['morphFeats_convexPerimeter'] = getConvexPerimeter(grayImg)
    features['morphFeats_equivalentDiameter'] = getEquivalentDiameter(grayImg)
    features['morphFeats_compactness'] = getCompactness(grayImg)
    features['morphFeats_convexity'] = getConvexity(grayImg)
    features['morphFeats_solidity'] = getSolidity(grayImg)
    #Shape Features (Ellipse)
    eccentricity, aspectRatio, majorAxis, minorAxis = getNucleosEccentricity(grayImg)
    features['morphFeats_nucleoEccentricity'] = eccentricity
    features['morphFeats_nucleoAspectRatio'] = aspectRatio
    features['morphFeats_nucleoMajorAxis'] = majorAxis
    features['morphFeats_nucleoMinorAxis'] = minorAxis
    features['morphFeats_nucleoElongation'] = getNucleusElongation(grayImg)
    features['morphFeats_rectangularity'] = getRectangularity(grayImg)
    
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
            featuresDict = getMorphologicalFeatures(imgPath)
            train_df = train_df.append(featuresDict, ignore_index=True)
            print(f'{n}-Train')
        cols = train_df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        train_df.to_csv(f'data/AugmPatLvDiv_TRAIN-MORPHData_{train_df.shape[1]-1}-Features_{train_df.shape[0]}-images.csv')

    #Process for Validation dataset
    def createValidDataframe():
        valid_df = pd.DataFrame()
        np.random.shuffle(VALID_IMGS)
        for n, imgFile in enumerate(VALID_IMGS):
            imgPath = augm_patLvDiv_valid / imgFile
            featuresDict = getMorphologicalFeatures(imgPath)
            valid_df = valid_df.append(featuresDict, ignore_index=True)
            print(f'{n}-Validation')
        cols = valid_df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        valid_df = valid_df[cols]
        valid_df.to_csv(f'data/AugmPatLvDiv_VALIDATION-MORPHData_{valid_df.shape[1]-1}-Features_{valid_df.shape[0]}-images.csv')
    
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
            featuresDict = getMorphologicalFeatures(imgPath)
            train_df = train_df.append(featuresDict, ignore_index=True)
            print(f'{n}-Train')
        cols = train_df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        train_df = train_df[cols]
        train_df.to_csv(f'data/AugmRndDiv_TRAIN-MORPHData_{train_df.shape[1]-1}-Features_{train_df.shape[0]}-images.csv')

    #Process for Validation dataset
    def createValidDataframe():
        valid_df = pd.DataFrame()
        np.random.shuffle(VALID_IMGS)
        for n, imgFile in enumerate(VALID_IMGS):
            imgPath = augm_rndDiv_valid/ imgFile
            featuresDict = getMorphologicalFeatures(imgPath)
            valid_df = valid_df.append(featuresDict, ignore_index=True)
            print(f'{n}-Validation')
        cols = valid_df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        valid_df = valid_df[cols]
        valid_df.to_csv(f'data/AugmRndDiv_VALIDATION-MORPHData_{valid_df.shape[1]-1}-Features_{valid_df.shape[0]}-images.csv')
    
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
            featuresDict = getMorphologicalFeatures(imgPath)
            train_df = train_df.append(featuresDict, ignore_index=True)
            print(f'{n}-Train')
        cols = train_df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        train_df = train_df[cols]
        train_df.to_csv(f'data/PatLvDiv_TRAIN-MORPHData_{train_df.shape[1]-1}-Features_{train_df.shape[0]}-images.csv')

    #Process for Validation dataset
    def createValidDataframe():
        valid_df = pd.DataFrame()
        np.random.shuffle(VALID_IMGS)
        for n, imgFile in enumerate(VALID_IMGS):
            featuresDict = {}
            imgPath = patLvDiv_valid / imgFile
            featuresDict = getMorphologicalFeatures(imgPath)
            valid_df = valid_df.append(featuresDict, ignore_index=True)
            print(f'{n}-Validation')
        cols = valid_df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        valid_df = valid_df[cols]
        valid_df.to_csv(f'data/PatLvDiv_VALIDATION-MORPHData_{valid_df.shape[1]-1}-Features_{valid_df.shape[0]}-images.csv')

    #Process for Test dataset
    def createTestDataframe():
        test_df = pd.DataFrame()
        np.random.shuffle(TEST_IMGS)
        for n, imgFile in enumerate(TEST_IMGS):
            featuresDict = {}
            imgPath = patLvDiv_test / imgFile
            featuresDict = getMorphologicalFeatures(imgPath)
            test_df = test_df.append(featuresDict, ignore_index=True)
            print(f'{n}-Test')
        cols = test_df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        test_df = test_df[cols]
        test_df.to_csv(f'data/PatLvDiv_TEST-MORPHData_{test_df.shape[1]-1}-Features_{test_df.shape[0]}-images.csv')
    
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
            featuresDict = getMorphologicalFeatures(imgPath)
            train_df = train_df.append(featuresDict, ignore_index=True)
            print(f'{n}-Train')
        cols = train_df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        train_df = train_df[cols]
        train_df.to_csv(f'data/rndDiv_TRAIN-MORPHData_{train_df.shape[1]-1}-Features_{train_df.shape[0]}-images.csv')

    #Process for Validation dataset
    def createValidDataframe():
        valid_df = pd.DataFrame()
        np.random.shuffle(VALID_IMGS)
        for n, imgFile in enumerate(VALID_IMGS):
            featuresDict = {}
            imgPath = rndDiv_valid / imgFile
            featuresDict = getMorphologicalFeatures(imgPath)
            valid_df = valid_df.append(featuresDict, ignore_index=True)
            print(f'{n}-Validation')
        cols = valid_df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        valid_df = valid_df[cols]
        valid_df.to_csv(f'data/rndDiv_VALIDATION-MORPHData_{valid_df.shape[1]-1}-Features_{valid_df.shape[0]}-images.csv')

    #Process for Test dataset
    def createTestDataframe():
        test_df = pd.DataFrame()
        np.random.shuffle(TEST_IMGS)
        for n, imgFile in enumerate(TEST_IMGS):
            featuresDict = {}
            imgPath = rndDiv_test / imgFile
            featuresDict = getMorphologicalFeatures(imgPath)
            test_df = test_df.append(featuresDict, ignore_index=True)
            print(f'{n}-Test')
        cols = test_df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        test_df = test_df[cols]
        test_df.to_csv(f'data/rndDiv_TEST-MORPHData_{test_df.shape[1]-1}-Features_{test_df.shape[0]}-images.csv')
    
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
    assert isinstance(cv2.imread(str(s0)), np.ndarray), 'Erro, check images path'
    assert len(getMorphologicalFeatures(s0)) == 21, 'Error, check feature extraction'
    #df = pd.read_csv('data/AugmRndDiv_TRAIN-MORPHData_20-Features_20000-images.csv')
    #val = df['morphFeats_nucleoMajorAxis'].values
    #print(np.mean(val))
    #print(np.std(val))
    
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
