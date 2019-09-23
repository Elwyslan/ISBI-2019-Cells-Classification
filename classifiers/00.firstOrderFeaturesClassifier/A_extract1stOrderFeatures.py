import os
import multiprocessing
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import radiomics
radiomics.setVerbosity(60)#Quiet
import SimpleITK as sitk

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

def readImage(path, color):
    if color=='rgb':
        return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    elif color=='gray':
        return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2GRAY)
    elif color=='hsv':
        return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2HSV)
    return None

def getPyRadImageAndMask(grayScaleImage, jointSeries=False):
    _,pyRadMask = cv2.threshold(grayScaleImage, 1, 1, cv2.THRESH_BINARY)
    pyRadMask = cv2.erode(pyRadMask, np.ones((2,2), np.uint8), iterations=1)
    pyRadimage = sitk.GetImageFromArray(grayScaleImage)
    pyRadMask = sitk.GetImageFromArray(pyRadMask)
    #Datails on 'jointSeries' in https://github.com/Radiomics/pyradiomics/issues/447
    if jointSeries:
        pyRadimage = sitk.JoinSeries(pyRadimage)
        pyRadMask = sitk.JoinSeries(pyRadMask)
    return pyRadimage, pyRadMask

def getFirstOrderFeatures(image):
    image, mask = getPyRadImageAndMask(image)
    rad = radiomics.firstorder.RadiomicsFirstOrder(image, mask)
    rad.execute()
    featuresDict = {}
    #Energy is a measure of the magnitude of voxel values in an image. A larger values implies a greater sum of the squares of these values.
    featuresDict['1stOrderFeats_energy'] = rad.getEnergyFeatureValue()
    #Total Energy is the value of Energy feature scaled by the volume of the voxel in cubic mm
    #featuresDict['total_Energy'] = rad.getTotalEnergyFeatureValue()
    #Entropy specifies the uncertainty/randomness in the image values. It measures the average amount of information required to encode the image values.
    featuresDict['1stOrderFeats_entropy'] = rad.getEntropyFeatureValue()
    #Minimum value
    featuresDict['1stOrderFeats_minimum_value'] = rad.getMinimumFeatureValue()
    #Maximum value
    featuresDict['1stOrderFeats_maximum_value'] = rad.getMaximumFeatureValue()
    #Range (Max - Min)
    featuresDict['1stOrderFeats_range'] = rad.getRangeFeatureValue()
    #90% of the data values lie below the 90th percentile
    featuresDict['1stOrderFeats_90th_percentile'] = rad.get90PercentileFeatureValue()
    #10% of the data values lie below the 10th percentile.
    featuresDict['1stOrderFeats_10th_percentile'] = rad.get10PercentileFeatureValue()
    #Difference between the 25th and 75th percentile of the image array
    featuresDict['1stOrderFeats_interquartile_range'] = rad.getInterquartileRangeFeatureValue()
    #The average gray level intensity within the ROI
    featuresDict['1stOrderFeats_mean'] = rad.getMeanFeatureValue()
    #Standard Deviation measures the amount of variation or dispersion from the Mean Value
    featuresDict['1stOrderFeats_standard_deviation'] = rad.getStandardDeviationFeatureValue()
    #Variance is the the mean of the squared distances of each intensity value from the Mean value. This is a measure of the spread of the distribution about the mean.
    featuresDict['1stOrderFeats_variance'] = rad.getVarianceFeatureValue()
    #The median gray level intensity within the ROI
    featuresDict['1stOrderFeats_median'] = rad.getMedianFeatureValue()
    #Mean Absolute Deviation is the mean distance of all intensity values from the Mean Value of the image array
    featuresDict['1stOrderFeats_mean_absolute_deviation'] = rad.getMeanAbsoluteDeviationFeatureValue()
    #Robust Mean Absolute Deviation is the mean distance of all intensity values from the Mean Value calculated on the subset of image array
    #with gray levels in between, or equal to the 10th and 90th percentile
    featuresDict['1stOrderFeats_robust_mean_absolute_deviation'] = rad.getRobustMeanAbsoluteDeviationFeatureValue()
    #RMS is the square-root of the mean of all the squared intensity values. It is another measure of the magnitude of the image values
    featuresDict['1stOrderFeats_root_mean_squared'] = rad.getRootMeanSquaredFeatureValue()
    #Skewness measures the asymmetry of the distribution of values about the Mean value.
    #Depending on where the tail is elongated and the mass of the distribution is concentrated, this value can be positive or negative
    featuresDict['1stOrderFeats_skewness'] = rad.getSkewnessFeatureValue()
    #Kurtosis is a measure of the ‘peakedness’ of the distribution of values in the image ROI.
    #A higher kurtosis implies that the mass of the distribution is concentrated towards the tail(s) rather than towards the mean.
    #A lower kurtosis implies the reverse: that the mass of the distribution is concentrated towards a spike near the Mean value
    featuresDict['1stOrderFeats_kurtosis'] = rad.getKurtosisFeatureValue()
    #Uniformity is a measure of the sum of the squares of each intensity value.
    #This is a measure of the homogeneity of the image array, where a greater uniformity implies a greater homogeneity or a smaller range of discrete intensity values.
    featuresDict['1stOrderFeats_uniformity'] = rad.getUniformityFeatureValue()
    return featuresDict

def extractFeatureDict(imgPath):
    featuresDict = {}
    if 'all.bmp' in imgPath.name:
        featuresDict['cellType(ALL=1, HEM=-1)'] = 1
    elif 'hem.bmp' in imgPath.name:
        featuresDict['cellType(ALL=1, HEM=-1)'] = -1
    else:
        return {}
    img = readImage(imgPath,'rgb')
    r, g, b = cv2.split(img)
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    feats0 = getFirstOrderFeatures(r)
    feats0 = {f'R_Chan_{k}': v for k, v in feats0.items()}
    feats1 = getFirstOrderFeatures(g)
    feats1 = {f'G_Chan_{k}': v for k, v in feats1.items()}
    feats2 = getFirstOrderFeatures(b)
    feats2 = {f'B_Chan_{k}': v for k, v in feats2.items()}
    feats3 = getFirstOrderFeatures(h)
    feats3 = {f'H_Chan_{k}': v for k, v in feats3.items()}
    feats4 = getFirstOrderFeatures(s)
    feats4 = {f'S_Chan_{k}': v for k, v in feats4.items()}
    feats5 = getFirstOrderFeatures(v)
    feats5 = {f'V_Chan_{k}': v for k, v in feats5.items()}
    featuresDict = {**featuresDict, **feats0, **feats1, **feats2, **feats3, **feats4, **feats5}
    return featuresDict

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
            featuresDict = {}
            imgPath = augm_patLvDiv_train / imgFile
            featuresDict = extractFeatureDict(imgPath)
            train_df = train_df.append(featuresDict, ignore_index=True)
            print(f'{n}-Train')
        cols = train_df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        train_df = train_df[cols]
        train_df.to_csv(f'data/AugmPatLvDiv_TRAIN-FOFData_{train_df.shape[1]-1}-Features_{train_df.shape[0]}-images.csv')

    #Process for Validation dataset
    def createValidDataframe():
        valid_df = pd.DataFrame()
        np.random.shuffle(VALID_IMGS)
        for n, imgFile in enumerate(VALID_IMGS):
            featuresDict = {}
            imgPath = augm_patLvDiv_valid / imgFile
            featuresDict = extractFeatureDict(imgPath)
            valid_df = valid_df.append(featuresDict, ignore_index=True)
            print(f'{n}-Validation')
        cols = valid_df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        valid_df = valid_df[cols]
        valid_df.to_csv(f'data/AugmPatLvDiv_VALIDATION-FOFData_{valid_df.shape[1]-1}-Features_{valid_df.shape[0]}-images.csv')
    
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
            featuresDict = {}
            imgPath = augm_rndDiv_train / imgFile
            featuresDict = extractFeatureDict(imgPath)
            train_df = train_df.append(featuresDict, ignore_index=True)
            print(f'{n}-Train')
        cols = train_df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        train_df = train_df[cols]
        train_df.to_csv(f'data/AugmRndDiv_TRAIN-FOFData_{train_df.shape[1]-1}-Features_{train_df.shape[0]}-images.csv')

    #Process for Validation dataset
    def createValidDataframe():
        valid_df = pd.DataFrame()
        np.random.shuffle(VALID_IMGS)
        for n, imgFile in enumerate(VALID_IMGS):
            featuresDict = {}
            imgPath = augm_rndDiv_valid/ imgFile
            featuresDict = extractFeatureDict(imgPath)
            valid_df = valid_df.append(featuresDict, ignore_index=True)
            print(f'{n}-Validation')
        cols = valid_df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        valid_df = valid_df[cols]
        valid_df.to_csv(f'data/AugmRndDiv_VALIDATION-FOFData_{valid_df.shape[1]-1}-Features_{valid_df.shape[0]}-images.csv')
    
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
            featuresDict = {}
            imgPath = patLvDiv_train / imgFile
            featuresDict = extractFeatureDict(imgPath)
            train_df = train_df.append(featuresDict, ignore_index=True)
            print(f'{n}-Train')
        cols = train_df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        train_df = train_df[cols]
        train_df.to_csv(f'data/PatLvDiv_TRAIN-FOFData_{train_df.shape[1]-1}-Features_{train_df.shape[0]}-images.csv')

    #Process for Validation dataset
    def createValidDataframe():
        valid_df = pd.DataFrame()
        np.random.shuffle(VALID_IMGS)
        for n, imgFile in enumerate(VALID_IMGS):
            featuresDict = {}
            imgPath = patLvDiv_valid / imgFile
            featuresDict = extractFeatureDict(imgPath)
            valid_df = valid_df.append(featuresDict, ignore_index=True)
            print(f'{n}-Validation')
        cols = valid_df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        valid_df = valid_df[cols]
        valid_df.to_csv(f'data/PatLvDiv_VALIDATION-FOFData_{valid_df.shape[1]-1}-Features_{valid_df.shape[0]}-images.csv')

    #Process for Test dataset
    def createTestDataframe():
        test_df = pd.DataFrame()
        np.random.shuffle(TEST_IMGS)
        for n, imgFile in enumerate(TEST_IMGS):
            featuresDict = {}
            imgPath = patLvDiv_test / imgFile
            featuresDict = extractFeatureDict(imgPath)
            test_df = test_df.append(featuresDict, ignore_index=True)
            print(f'{n}-Test')
        cols = test_df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        test_df = test_df[cols]
        test_df.to_csv(f'data/PatLvDiv_TEST-FOFData_{test_df.shape[1]-1}-Features_{test_df.shape[0]}-images.csv')
    
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
            featuresDict = extractFeatureDict(imgPath)
            train_df = train_df.append(featuresDict, ignore_index=True)
            print(f'{n}-Train')
        cols = train_df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        train_df = train_df[cols]
        train_df.to_csv(f'data/rndDiv_TRAIN-FOFData_{train_df.shape[1]-1}-Features_{train_df.shape[0]}-images.csv')

    #Process for Validation dataset
    def createValidDataframe():
        valid_df = pd.DataFrame()
        np.random.shuffle(VALID_IMGS)
        for n, imgFile in enumerate(VALID_IMGS):
            featuresDict = {}
            imgPath = rndDiv_valid / imgFile
            featuresDict = extractFeatureDict(imgPath)
            valid_df = valid_df.append(featuresDict, ignore_index=True)
            print(f'{n}-Validation')
        cols = valid_df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        valid_df = valid_df[cols]
        valid_df.to_csv(f'data/rndDiv_VALIDATION-FOFData_{valid_df.shape[1]-1}-Features_{valid_df.shape[0]}-images.csv')

    #Process for Test dataset
    def createTestDataframe():
        test_df = pd.DataFrame()
        np.random.shuffle(TEST_IMGS)
        for n, imgFile in enumerate(TEST_IMGS):
            featuresDict = {}
            imgPath = rndDiv_test / imgFile
            featuresDict = extractFeatureDict(imgPath)
            test_df = test_df.append(featuresDict, ignore_index=True)
            print(f'{n}-Test')
        cols = test_df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        test_df = test_df[cols]
        test_df.to_csv(f'data/rndDiv_TEST-FOFData_{test_df.shape[1]-1}-Features_{test_df.shape[0]}-images.csv')
    
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
    assert isinstance(readImage(s0, 'rgb'), np.ndarray), 'Erro, check images path'
    assert len(getFirstOrderFeatures(readImage(s0,'gray'))) == 18, 'Error, check feature extraction'
    assert len(extractFeatureDict(s0)) == 109, 'Error, check dataframe creation'

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
