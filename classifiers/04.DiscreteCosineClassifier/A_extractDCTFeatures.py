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

#https://github.com/getsanjeev/compression-DCT/blob/master/zigzag.py
def inverse_zigzag(inputArray, vmax, hmax):
    h, v, vmin, hmin = 0, 0, 0, 0
    output = np.zeros((vmax, hmax))
    i = 0
    while ((v < vmax) and (h < hmax)):
        if ((h + v) % 2) == 0:
            if (v == vmin):
                output[v, h] = inputArray[i]
                if (h == hmax):
                    v = v + 1
                else:
                    h = h + 1
                i = i + 1
            elif ((h == hmax -1 ) and (v < vmax)):
                output[v, h] = inputArray[i]
                v = v + 1
                i = i + 1
            elif ((v > vmin) and (h < hmax -1 )):
                output[v, h] = inputArray[i] 
                v = v - 1
                h = h + 1
                i = i + 1
        else:
            if ((v == vmax -1) and (h <= hmax -1)):
                output[v, h] = inputArray[i] 
                h = h + 1
                i = i + 1
            elif (h == hmin):
                output[v, h] = inputArray[i] 
                if (v == vmax -1):
                    h = h + 1
                else:
                    v = v + 1
                i = i + 1
            elif((v < vmax -1) and (h > hmin)):
                output[v, h] = inputArray[i] 
                v = v + 1
                h = h - 1
                i = i + 1
        if ((v == vmax-1) and (h == hmax-1)):
            output[v, h] = inputArray[i] 
            break
    return output


def computeDCTransform(imgPath, noOfFeatures=1024):
    srcImg = cv2.imread(str(imgPath))
    srcImg = cv2.cvtColor(srcImg, cv2.COLOR_BGR2GRAY)
    #Compute DCT transform
    srcImg = np.array(srcImg, dtype=np.float32)
    dctFeats = cv2.dct(srcImg)
    #Thank's to https://stackoverflow.com/questions/50445847/how-to-zigzag-order-and-concatenate-the-value-every-line-using-python
    zigzagPattern = np.concatenate([np.diagonal(dctFeats[::-1,:], i)[::(2*(i % 2)-1)] for i in range(1-dctFeats.shape[0], dctFeats.shape[0])])
    zigzagPattern = zigzagPattern[0:noOfFeatures]
    #plt.bar([i for i in range(len(zigzagPattern))], zigzagPattern)
    #plt.plot(dctFeats, '*r')
    #plt.show()
    return zigzagPattern

def getDCTFeatures(imagePath):
    features = {}
    if 'all.bmp' in imagePath.name:
        features['cellType(ALL=1, HEM=-1)'] = 1
    elif 'hem.bmp' in imagePath.name:
        features['cellType(ALL=1, HEM=-1)'] = -1
    else:
        return {}
    feats = computeDCTransform(imagePath)
    for i in range(len(feats)):
        features[f'pt{str(i).zfill(4)}'] = feats[i]
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
            featuresDict = getDCTFeatures(imgPath)
            train_df = train_df.append(featuresDict, ignore_index=True)
            print(f'{n}-Train')
        cols = train_df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        train_df.to_csv(f'data/AugmPatLvDiv_TRAIN-DCTData_{train_df.shape[1]-1}-Features_{train_df.shape[0]}-images.csv')

    #Process for Validation dataset
    def createValidDataframe():
        valid_df = pd.DataFrame()
        np.random.shuffle(VALID_IMGS)
        for n, imgFile in enumerate(VALID_IMGS):
            imgPath = augm_patLvDiv_valid / imgFile
            featuresDict = getDCTFeatures(imgPath)
            valid_df = valid_df.append(featuresDict, ignore_index=True)
            print(f'{n}-Validation')
        cols = valid_df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        valid_df = valid_df[cols]
        valid_df.to_csv(f'data/AugmPatLvDiv_VALIDATION-DCTData_{valid_df.shape[1]-1}-Features_{valid_df.shape[0]}-images.csv')
    
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
            featuresDict = getDCTFeatures(imgPath)
            train_df = train_df.append(featuresDict, ignore_index=True)
            print(f'{n}-Train')
        cols = train_df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        train_df = train_df[cols]
        train_df.to_csv(f'data/AugmRndDiv_TRAIN-DCTData_{train_df.shape[1]-1}-Features_{train_df.shape[0]}-images.csv')

    #Process for Validation dataset
    def createValidDataframe():
        valid_df = pd.DataFrame()
        np.random.shuffle(VALID_IMGS)
        for n, imgFile in enumerate(VALID_IMGS):
            imgPath = augm_rndDiv_valid/ imgFile
            featuresDict = getDCTFeatures(imgPath)
            valid_df = valid_df.append(featuresDict, ignore_index=True)
            print(f'{n}-Validation')
        cols = valid_df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        valid_df = valid_df[cols]
        valid_df.to_csv(f'data/AugmRndDiv_VALIDATION-DCTData_{valid_df.shape[1]-1}-Features_{valid_df.shape[0]}-images.csv')
    
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
            featuresDict = getDCTFeatures(imgPath)
            train_df = train_df.append(featuresDict, ignore_index=True)
            print(f'{n}-Train')
        cols = train_df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        train_df = train_df[cols]
        train_df.to_csv(f'data/PatLvDiv_TRAIN-DCTData_{train_df.shape[1]-1}-Features_{train_df.shape[0]}-images.csv')

    #Process for Validation dataset
    def createValidDataframe():
        valid_df = pd.DataFrame()
        np.random.shuffle(VALID_IMGS)
        for n, imgFile in enumerate(VALID_IMGS):
            featuresDict = {}
            imgPath = patLvDiv_valid / imgFile
            featuresDict = getDCTFeatures(imgPath)
            valid_df = valid_df.append(featuresDict, ignore_index=True)
            print(f'{n}-Validation')
        cols = valid_df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        valid_df = valid_df[cols]
        valid_df.to_csv(f'data/PatLvDiv_VALIDATION-DCTData_{valid_df.shape[1]-1}-Features_{valid_df.shape[0]}-images.csv')

    #Process for Test dataset
    def createTestDataframe():
        test_df = pd.DataFrame()
        np.random.shuffle(TEST_IMGS)
        for n, imgFile in enumerate(TEST_IMGS):
            featuresDict = {}
            imgPath = patLvDiv_test / imgFile
            featuresDict = getDCTFeatures(imgPath)
            test_df = test_df.append(featuresDict, ignore_index=True)
            print(f'{n}-Test')
        cols = test_df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        test_df = test_df[cols]
        test_df.to_csv(f'data/PatLvDiv_TEST-DCTData_{test_df.shape[1]-1}-Features_{test_df.shape[0]}-images.csv')
    
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
            featuresDict = getDCTFeatures(imgPath)
            train_df = train_df.append(featuresDict, ignore_index=True)
            print(f'{n}-Train')
        cols = train_df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        train_df = train_df[cols]
        train_df.to_csv(f'data/rndDiv_TRAIN-DCTData_{train_df.shape[1]-1}-Features_{train_df.shape[0]}-images.csv')

    #Process for Validation dataset
    def createValidDataframe():
        valid_df = pd.DataFrame()
        np.random.shuffle(VALID_IMGS)
        for n, imgFile in enumerate(VALID_IMGS):
            featuresDict = {}
            imgPath = rndDiv_valid / imgFile
            featuresDict = getDCTFeatures(imgPath)
            valid_df = valid_df.append(featuresDict, ignore_index=True)
            print(f'{n}-Validation')
        cols = valid_df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        valid_df = valid_df[cols]
        valid_df.to_csv(f'data/rndDiv_VALIDATION-DCTData_{valid_df.shape[1]-1}-Features_{valid_df.shape[0]}-images.csv')

    #Process for Test dataset
    def createTestDataframe():
        test_df = pd.DataFrame()
        np.random.shuffle(TEST_IMGS)
        for n, imgFile in enumerate(TEST_IMGS):
            featuresDict = {}
            imgPath = rndDiv_test / imgFile
            featuresDict = getDCTFeatures(imgPath)
            test_df = test_df.append(featuresDict, ignore_index=True)
            print(f'{n}-Test')
        cols = test_df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        test_df = test_df[cols]
        test_df.to_csv(f'data/rndDiv_TEST-DCTData_{test_df.shape[1]-1}-Features_{test_df.shape[0]}-images.csv')
    
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
    #s0 = Path('../../data/augm_rndDiv_train/AugmentedImg_1051_UID_H24_33_11_hem.bmp')
    #s0 = Path('../../train/fold_2/hem/UID_h3_3_3_hem.bmp')
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
