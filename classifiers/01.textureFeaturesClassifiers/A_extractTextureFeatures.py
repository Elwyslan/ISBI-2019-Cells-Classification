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
    #pyRadMask = cv2.dilate(pyRadMask, np.ones((2,2), np.uint8), iterations=1)
    pyRadimage = sitk.GetImageFromArray(grayScaleImage)
    pyRadMask = sitk.GetImageFromArray(pyRadMask)
    #Datails on 'jointSeries' in https://github.com/Radiomics/pyradiomics/issues/447
    if jointSeries:
        pyRadimage = sitk.JoinSeries(pyRadimage)
        pyRadMask = sitk.JoinSeries(pyRadMask)
    return pyRadimage, pyRadMask

def getGLCMFeatures(image):
    image, mask = getPyRadImageAndMask(image, jointSeries=True)
    rad = radiomics.glcm.RadiomicsGLCM(image, mask)
    rad.execute()
    featuresDict = {}
    #Autocorrelation is a measure of the magnitude of the fineness and coarseness of texture
    featuresDict['GLCM_autocorrelation'] = rad.getAutocorrelationFeatureValue()
    #Returns the mean gray level intensity of the i distribution.
    featuresDict['GLCM_joint_average'] = rad.getJointAverageFeatureValue()
    #Cluster Prominence is a measure of the skewness and asymmetry of the GLCM.
    #A higher values implies more asymmetry about the mean while a lower value indicates a peak near the mean value and less variation about the mean
    featuresDict['GLCM_cluster_prominence'] = rad.getClusterProminenceFeatureValue()
    #Cluster Shade is a measure of the skewness and uniformity of the GLCM.
    #A higher cluster shade implies greater asymmetry about the mean.
    featuresDict['GLCM_cluster_shade'] = rad.getClusterShadeFeatureValue()
    #Cluster Tendency is a measure of groupings of voxels with similar gray-level values
    featuresDict['GLCM_cluster_tendency'] = rad.getClusterTendencyFeatureValue()
    #Contrast is a measure of the local intensity variation, favoring values away from the diagonal (i=j).
    #A larger value correlates with a greater disparity in intensity values among neighboring voxels.
    featuresDict['GLCM_contrast'] = rad.getContrastFeatureValue()
    #Correlation is a value between 0 (uncorrelated) and 1 (perfectly correlated) 
    #showing the linear dependency of gray level values to their respective voxels in the GLCM.
    featuresDict['GLCM_correlation'] = rad.getCorrelationFeatureValue()
    #Difference Average measures the relationship between occurrences of pairs with similar intensity values and occurrences of pairs with differing intensity values.
    featuresDict['GLCM_difference_average'] = rad.getDifferenceAverageFeatureValue()
    #Difference Entropy is a measure of the randomness/variability in neighborhood intensity value differences.
    featuresDict['GLCM_difference_entropy'] = rad.getDifferenceEntropyFeatureValue()
    #Difference Variance is a measure of heterogeneity that places higher weights on differing intensity level pairs that deviate more from the mean.
    featuresDict['GLCM_difference_variance'] = rad.getDifferenceVarianceFeatureValue()
    #Energy is a measure of homogeneous patterns in the image.
    #A greater Energy implies that there are more instances of intensity value pairs in the image that neighbor each other at higher frequencies.
    featuresDict['GLCM_joint_energy'] = rad.getJointEnergyFeatureValue()
    #Joint entropy is a measure of the randomness/variability in neighborhood intensity values
    featuresDict['GLCM_joint_entropy'] = rad.getJointEntropyFeatureValue()
    #IMC1 assesses the correlation between the probability distributions of i and j (quantifying the complexity of the texture), using mutual information I(x, y):
    featuresDict['GLCM_informational_measure_correlation_1'] = rad.getImc1FeatureValue()
    #IMC2 also assesses the correlation between the probability distributions of i and j (quantifying the complexity of the texture)
    featuresDict['GLCM_informational_measure_correlation_2'] = rad.getImc2FeatureValue()
    #IDM (a.k.a Homogeneity 2) is a measure of the local homogeneity of an image. IDM weights are the inverse of the Contrast weights
    featuresDict['GLCM_inverse_difference_moment'] = rad.getIdmFeatureValue()
    #The Maximal Correlation Coefficient is a measure of complexity of the texture and 0≤MCC≤1.
    featuresDict['GLCM_maximal_correlation_coefficient'] = rad.getMCCFeatureValue()
    #IDMN (inverse difference moment normalized) is a measure of the local homogeneity of an image.
    featuresDict['GLCM_inverse_difference_moment_normalized'] = rad.getIdmnFeatureValue()
    #ID (a.k.a. Homogeneity 1) is another measure of the local homogeneity of an image.
    #With more uniform gray levels, the denominator will remain low, resulting in a higher overall value
    featuresDict['GLCM_inverse_difference'] = rad.getIdFeatureValue()
    #DN (inverse difference normalized) is another measure of the local homogeneity of an image.
    #Unlike Homogeneity1, IDN normalizes the difference between the neighboring intensity values by dividing over the total number of discrete intensity values
    featuresDict['GLCM_inverse_difference_normalized'] = rad.getIdnFeatureValue()
    featuresDict['GLCM_inverse_variance'] = rad.getInverseVarianceFeatureValue()
    #Maximum Probability is occurrences of the most predominant pair of neighboring intensity values
    featuresDict['GLCM_maximum_probability'] = rad.getMaximumProbabilityFeatureValue()
    #Sum Average measures the relationship between occurrences of pairs with lower intensity values and occurrences of pairs with higher intensity values
    featuresDict['GLCM_sum_average'] = rad.getSumAverageFeatureValue()
    #Sum Entropy is a sum of neighborhood intensity value differences
    featuresDict['GLCM_sum_entropy'] = rad.getSumEntropyFeatureValue()
    #Sum of Squares or Variance is a measure in the distribution of neigboring intensity level pairs about the mean intensity level in the GLCM
    featuresDict['GLCM_sum_of_squares'] = rad.getSumSquaresFeatureValue()
    return featuresDict

def getGLDMFeatures(image):
    image, mask = getPyRadImageAndMask(image, jointSeries=True)
    rad = radiomics.gldm.RadiomicsGLDM(image, mask)
    rad.execute()
    featuresDict = {}
    #A measure of the distribution of small dependencies, with a greater value indicative of smaller dependence and less homogeneous textures.
    featuresDict['GLDM_small_dependence_emphasis'] = rad.getSmallDependenceEmphasisFeatureValue()
    #A measure of the distribution of large dependencies, with a greater value indicative of larger dependence and more homogeneous textures.
    featuresDict['GLDM_large_dependence_emphasis'] = rad.getLargeDependenceEmphasisFeatureValue()
    #Measures the similarity of gray-level intensity values in the image, where a lower GLN value correlates with a greater similarity in intensity values.
    featuresDict['GLDM_gray_level_nonUniformity'] = rad.getGrayLevelNonUniformityFeatureValue()
    #Measures the similarity of dependence throughout the image, with a lower value indicating more homogeneity among dependencies in the image.
    featuresDict['GLDM_dependence_nonUniformity'] = rad.getDependenceNonUniformityFeatureValue()
    #Measures the similarity of dependence throughout the image,
    #with a lower value indicating more homogeneity among dependencies in the image. This is the normalized version of the DLN formula.
    featuresDict['GLDM_dependence_nonUniformity_normalized'] = rad.getDependenceNonUniformityNormalizedFeatureValue()
    #Measures the variance in grey level in the image.
    featuresDict['GLDM_gray_level_variance'] = rad.getGrayLevelVarianceFeatureValue()
    #Measures the variance in dependence size in the image.
    featuresDict['GLDM_dependence_variance'] = rad.getDependenceVarianceFeatureValue()
    #Measures the variance in dependence size in the image.
    featuresDict['GLDM_dependence_entropy'] = rad.getDependenceEntropyFeatureValue()
    #Measures the distribution of low gray-level values, with a higher value indicating a greater concentration of low gray-level values in the image.
    featuresDict['GLDM_low_gray_level_emphasis'] = rad.getLowGrayLevelEmphasisFeatureValue()
    #Measures the distribution of the higher gray-level values, with a higher value indicating a greater concentration of high gray-level values in the image.
    featuresDict['GLDM_high_gray_level_emphasis'] = rad.getHighGrayLevelEmphasisFeatureValue()
    #Measures the joint distribution of small dependence with lower gray-level values.
    featuresDict['GLDM_small_dependence_low_gray_level_emphasis'] = rad.getSmallDependenceLowGrayLevelEmphasisFeatureValue()
    #Measures the joint distribution of small dependence with higher gray-level values.
    featuresDict['GLDM_small_dependence_high_gray_level_emphasis'] = rad.getSmallDependenceHighGrayLevelEmphasisFeatureValue()
    #Measures the joint distribution of large dependence with lower gray-level values.
    featuresDict['GLDM_large_dependence_low_gray_level_emphasis'] = rad.getLargeDependenceLowGrayLevelEmphasisFeatureValue()
    #Measures the joint distribution of large dependence with higher gray-level values.
    featuresDict['GLDM_large_dependence_high_gray_level_emphasis'] = rad.getLargeDependenceHighGrayLevelEmphasisFeatureValue()
    return featuresDict

def getGLRLMFeatures(image):
    image, mask = getPyRadImageAndMask(image, jointSeries=True)
    rad = radiomics.glrlm.RadiomicsGLRLM(image, mask)
    rad.execute()
    featuresDict = {}
    #SRE is a measure of the distribution of short run lengths, with a greater value indicative of shorter run lengths and more fine textural textures.
    featuresDict['GLRLM_short_run_emphasis'] = rad.getShortRunEmphasisFeatureValue()
    #LRE is a measure of the distribution of long run lengths, with a greater value indicative of longer run lengths and more coarse structural textures.
    featuresDict['GLRLM_long_run_emphasis'] = rad.getLongRunEmphasisFeatureValue()
    #GLN measures the similarity of gray-level intensity values in the image, where a lower GLN value correlates with a greater similarity in intensity values.
    featuresDict['GLRLM_gray_level_nonUniformity'] = rad.getGrayLevelNonUniformityFeatureValue()
    #GLNN measures the similarity of gray-level intensity values in the image,
    #where a lower GLNN value correlates with a greater similarity in intensity values. This is the normalized version of the GLN formula.
    featuresDict['GLRLM_gray_level_nonUniformity_normalized'] = rad.getGrayLevelNonUniformityNormalizedFeatureValue()
    #RLN measures the similarity of run lengths throughout the image,
    #with a lower value indicating more homogeneity among run lengths in the image.
    featuresDict['GLRLM_run_length_nonUniformity'] = rad.getRunLengthNonUniformityFeatureValue()
    #RLNN measures the similarity of run lengths throughout the image,
    #with a lower value indicating more homogeneity among run lengths in the image. This is the normalized version of the RLN formula.
    featuresDict['GLRLM_run_length_nonUniformity_normalized'] = rad.getRunLengthNonUniformityNormalizedFeatureValue()
    #RP measures the coarseness of the texture by taking the ratio of number of runs and number of voxels in the ROI.
    featuresDict['GLRLM_run_percentage'] = rad.getRunPercentageFeatureValue()
    #GLV measures the variance in gray level intensity for the runs.
    featuresDict['GLRLM_gray_level_variance'] = rad.getGrayLevelVarianceFeatureValue()
    #RV is a measure of the variance in runs for the run lengths.
    featuresDict['GLRLM_run_variance'] = rad.getRunVarianceFeatureValue()
    #RE measures the uncertainty/randomness in the distribution of run lengths and gray levels. A higher value indicates more heterogeneity in the texture patterns.
    featuresDict['GLRLM_run_entropy'] = rad.getRunEntropyFeatureValue()
    #LGLRE measures the distribution of low gray-level values, with a higher value indicating a greater concentration of low gray-level values in the image.
    featuresDict['GLRLM_low_gray_level_run_emphasis'] = rad.getLowGrayLevelRunEmphasisFeatureValue()
    #HGLRE measures the distribution of the higher gray-level values, with a higher value indicating a greater concentration of high gray-level values in the image.
    featuresDict['GLRLM_high_gray_level_run_emphasis'] = rad.getHighGrayLevelRunEmphasisFeatureValue()
    #SRLGLE measures the joint distribution of shorter run lengths with lower gray-level values.
    featuresDict['GLRLM_short_run_low_gray_level_emphasis'] = rad.getShortRunLowGrayLevelEmphasisFeatureValue()
    #SRHGLE measures the joint distribution of shorter run lengths with higher gray-level values.
    featuresDict['GLRLM_short_run_high_gray_level_emphasis'] = rad.getShortRunHighGrayLevelEmphasisFeatureValue()
    #LRLGLRE measures the joint distribution of long run lengths with lower gray-level values.
    featuresDict['GLRLM_long_run_low_gray_level_emphasis'] = rad.getLongRunLowGrayLevelEmphasisFeatureValue()
    #LRHGLRE measures the joint distribution of long run lengths with higher gray-level values.
    featuresDict['GLRLM_long_run_high_gray_level_emphasis'] = rad.getLongRunHighGrayLevelEmphasisFeatureValue()
    return featuresDict

def getGLSZMFeatures(image):
    image, mask = getPyRadImageAndMask(image, jointSeries=True)
    rad = radiomics.glszm.RadiomicsGLSZM(image, mask)
    rad.execute()
    featuresDict = {}
    #SAE is a measure of the distribution of small size zones, with a greater value indicative of more smaller size zones and more fine textures
    featuresDict['GLSZM_small_area_emphasis'] = rad.getSmallAreaEmphasisFeatureValue()
    #LAE is a measure of the distribution of large area size zones, with a greater value indicative of more larger size zones and more coarse textures.
    featuresDict['GLSZM_large_area_emphasis'] = rad.getLargeAreaEmphasisFeatureValue()
    #GLN measures the variability of gray-level intensity values in the image, with a lower value indicating more homogeneity in intensity values.
    featuresDict['GLSZM_gray_level_nonUniformity'] = rad.getGrayLevelNonUniformityFeatureValue()
    #GLNN measures the variability of gray-level intensity values in the image,
    #with a lower value indicating a greater similarity in intensity values. This is the normalized version of the GLN formula.
    featuresDict['GLSZM_gray_level_nonUniformity_normalized'] = rad.getGrayLevelNonUniformityNormalizedFeatureValue()
    #SZN measures the variability of size zone volumes in the image, with a lower value indicating more homogeneity in size zone volumes.
    featuresDict['GLSZM_size_zone_nonUniformity'] = rad.getSizeZoneNonUniformityFeatureValue()
    #SZNN measures the variability of size zone volumes throughout the image, with a lower value indicating more homogeneity among zone size volumes in the image.
    #This is the normalized version of the SZN formula.
    featuresDict['GLSZM_size_zone_nonUniformity_normalized'] = rad.getSizeZoneNonUniformityNormalizedFeatureValue()
    #ZP measures the coarseness of the texture by taking the ratio of number of zones and number of voxels in the ROI.
    featuresDict['GLSZM_zone_percentage'] = rad.getZonePercentageFeatureValue()
    #GLV measures the variance in gray level intensities for the zones.
    featuresDict['GLSZM_gray_level_variance'] = rad.getGrayLevelVarianceFeatureValue()
    #ZV measures the variance in zone size volumes for the zones.
    featuresDict['GLSZM_zone_variance'] = rad.getZoneVarianceFeatureValue()
    #ZE measures the uncertainty/randomness in the distribution of zone sizes and gray levels. A higher value indicates more heterogeneneity in the texture patterns.
    featuresDict['GLSZM_zone_entropy'] = rad.getZoneEntropyFeatureValue()
    #LGLZE measures the distribution of lower gray-level size zones, with a higher value indicating a greater proportion of lower gray-level values and size zones in the image.
    featuresDict['GLSZM_low_gray_level_zone_emphasis'] = rad.getLowGrayLevelZoneEmphasisFeatureValue()
    #HGLZE measures the distribution of the higher gray-level values, with a higher value indicating a greater proportion of higher gray-level values and size zones in the image.
    featuresDict['GLSZM_high_gray_level_zone_emphasis'] = rad.getHighGrayLevelZoneEmphasisFeatureValue()
    #SALGLE measures the proportion in the image of the joint distribution of smaller size zones with lower gray-level values.
    featuresDict['GLSZM_small_area_low_gray_level_emphasis'] = rad.getSmallAreaLowGrayLevelEmphasisFeatureValue()
    #SAHGLE measures the proportion in the image of the joint distribution of smaller size zones with higher gray-level values.
    featuresDict['GLSZM_small_area_high_gray_level_emphasis'] = rad.getSmallAreaHighGrayLevelEmphasisFeatureValue()
    #LALGLE measures the proportion in the image of the joint distribution of larger size zones with lower gray-level values.
    featuresDict['GLSZM_large_area_low_gray_level_emphasis'] = rad.getLargeAreaLowGrayLevelEmphasisFeatureValue()
    #LAHGLE measures the proportion in the image of the joint distribution of larger size zones with higher gray-level values.
    featuresDict['GLSZM_large_area_high_gray_level_emphasis'] = rad.getLargeAreaHighGrayLevelEmphasisFeatureValue()
    return featuresDict

def getNGTDMFeatures(image):
    image, mask = getPyRadImageAndMask(image, jointSeries=True)
    rad = radiomics.ngtdm.RadiomicsNGTDM(image, mask)
    rad.execute()
    featuresDict = {}
    #Coarseness is a measure of average difference between the center voxel and its neighbourhood
    #and is an indication of the spatial rate of change. A higher value indicates a lower spatial change rate and a locally more uniform texture.
    featuresDict['NGTDM_coarseness'] = rad.getCoarsenessFeatureValue()
    #Contrast is a measure of the spatial intensity change, but is also dependent on the overall gray level dynamic range.
    #Contrast is high when both the dynamic range and the spatial change rate are high, i.e.
    #an image with a large range of gray levels, with large changes between voxels and their neighbourhood.
    featuresDict['NGTDM_contrast'] = rad.getContrastFeatureValue()
    #A measure of the change from a pixel to its neighbour.
    #A high value for busyness indicates a ‘busy’ image, with rapid changes of intensity between pixels and its neighbourhood.
    featuresDict['NGTDM_busyness'] = rad.getBusynessFeatureValue()
    #An image is considered complex when there are many primitive components in the image, i.e.
    #the image is non-uniform and there are many rapid changes in gray level intensity.
    featuresDict['NGTDM_complexity'] = rad.getComplexityFeatureValue()
    #Strenght is a measure of the primitives in an image.
    #Its value is high when the primitives are easily defined and visible, i.e.
    #an image with slow change in intensity but more large coarse differences in gray level intensities.
    featuresDict['NGTDM_strength'] = rad.getStrengthFeatureValue()
    return featuresDict

def extractFeatureDict(imgPath):
    featuresDict = {}
    if 'all.bmp' in imgPath.name:
        featuresDict['cellType(ALL=1, HEM=-1)'] = 1
    elif 'hem.bmp' in imgPath.name:
        featuresDict['cellType(ALL=1, HEM=-1)'] = -1
    else:
        return {}
    img = readImage(imgPath,'gray')
    featuresDict = {**featuresDict, **getGLCMFeatures(img), **getGLDMFeatures(img),
                    **getGLRLMFeatures(img), **getGLSZMFeatures(img),
                    **getNGTDMFeatures(img)}
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
            imgPath = augm_patLvDiv_train / imgFile
            featuresDict = extractFeatureDict(imgPath)
            train_df = train_df.append(featuresDict, ignore_index=True)
            print(f'{n}-Train')
        cols = train_df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        train_df.to_csv(f'data/AugmPatLvDiv_TRAIN-TEXTUREData_{train_df.shape[1]-1}-Features_{train_df.shape[0]}-images.csv')

    #Process for Validation dataset
    def createValidDataframe():
        valid_df = pd.DataFrame()
        np.random.shuffle(VALID_IMGS)
        for n, imgFile in enumerate(VALID_IMGS):
            imgPath = augm_patLvDiv_valid / imgFile
            featuresDict = extractFeatureDict(imgPath)
            valid_df = valid_df.append(featuresDict, ignore_index=True)
            print(f'{n}-Validation')
        cols = valid_df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        valid_df = valid_df[cols]
        valid_df.to_csv(f'data/AugmPatLvDiv_VALIDATION-TEXTUREData_{valid_df.shape[1]-1}-Features_{valid_df.shape[0]}-images.csv')
    
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
            featuresDict = extractFeatureDict(imgPath)
            train_df = train_df.append(featuresDict, ignore_index=True)
            print(f'{n}-Train')
        cols = train_df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        train_df = train_df[cols]
        train_df.to_csv(f'data/AugmRndDiv_TRAIN-TEXTUREData_{train_df.shape[1]-1}-Features_{train_df.shape[0]}-images.csv')

    #Process for Validation dataset
    def createValidDataframe():
        valid_df = pd.DataFrame()
        np.random.shuffle(VALID_IMGS)
        for n, imgFile in enumerate(VALID_IMGS):
            imgPath = augm_rndDiv_valid/ imgFile
            featuresDict = extractFeatureDict(imgPath)
            valid_df = valid_df.append(featuresDict, ignore_index=True)
            print(f'{n}-Validation')
        cols = valid_df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        valid_df = valid_df[cols]
        valid_df.to_csv(f'data/AugmRndDiv_VALIDATION-TEXTUREData_{valid_df.shape[1]-1}-Features_{valid_df.shape[0]}-images.csv')
    
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
            featuresDict = extractFeatureDict(imgPath)
            train_df = train_df.append(featuresDict, ignore_index=True)
            print(f'{n}-Train')
        cols = train_df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        train_df = train_df[cols]
        train_df.to_csv(f'data/PatLvDiv_TRAIN-TEXTUREData_{train_df.shape[1]-1}-Features_{train_df.shape[0]}-images.csv')

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
        valid_df.to_csv(f'data/PatLvDiv_VALIDATION-TEXTUREData_{valid_df.shape[1]-1}-Features_{valid_df.shape[0]}-images.csv')

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
        test_df.to_csv(f'data/PatLvDiv_TEST-TEXTUREData_{test_df.shape[1]-1}-Features_{test_df.shape[0]}-images.csv')
    
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
        train_df.to_csv(f'data/rndDiv_TRAIN-TEXTUREData_{train_df.shape[1]-1}-Features_{train_df.shape[0]}-images.csv')

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
        valid_df.to_csv(f'data/rndDiv_VALIDATION-TEXTUREData_{valid_df.shape[1]-1}-Features_{valid_df.shape[0]}-images.csv')

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
        test_df.to_csv(f'data/rndDiv_TEST-TEXTUREData_{test_df.shape[1]-1}-Features_{test_df.shape[0]}-images.csv')
    
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
    assert isinstance(readImage(s0,'gray'), np.ndarray), 'Erro, check images path'
    assert len(getGLCMFeatures(readImage(s0,'gray')))==24, 'Erro, check GLCM features'
    assert len(getGLRLMFeatures(readImage(s0,'gray')))==16, 'Erro, check GLRLM features'
    assert len(getGLSZMFeatures(readImage(s0,'gray')))==16, 'Erro, check GLSZM features'
    assert len(getNGTDMFeatures(readImage(s0,'gray')))==5, 'Erro, check NGTDM features'
    assert len(getGLDMFeatures(readImage(s0,'gray')))==14, 'Erro, check GLDM features'
    assert len(extractFeatureDict(s0))==76, 'Erro, check dataframe creation'

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
