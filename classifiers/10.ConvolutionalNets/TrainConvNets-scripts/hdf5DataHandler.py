import os
import cv2
import numpy as np
import h5py
from skimage import exposure

DATASET = 'augmRndDiv'
TRAIN_DATA_PATH = 'data/augm_rndDiv_train'
VALIDATION_DATA_PATH = 'data/augm_rndDiv_valid'

ALL_LABEL = [[1.0, 0.0]]
HEM_LABEL = [[0.0, 1.0]]

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
        print('Max:',np.max(img[:,:,0]), np.max(img[:,:,1]), np.max(img[:,:,2]))
        print('Min:',np.min(img[:,:,0]), np.min(img[:,:,1]), np.min(img[:,:,2]))
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

def createHDF5_validation(noOfImages, subtractMean=True, divideStdDev=True, colorScheme='rgb', imgSize=(450,450)):
    #Read Images
    images = os.listdir(VALIDATION_DATA_PATH)
    np.random.shuffle(images)
    
    #Drop images until desired size
    droppedALL = droppedHEM = 0
    dropQtd = np.ceil((len(images) - noOfImages)/2)
    while len(images)>noOfImages:
        if ('hem.bmp' in images[0]) and (droppedHEM<dropQtd):
            droppedHEM+=1
            images.pop(0)
        if ('all.bmp' in images[0]) and (droppedALL<dropQtd):
            droppedALL+=1
            images.pop(0)
        np.random.shuffle(images)
        print('ALL: {}, HEM: {}'.format(droppedALL, droppedHEM))
    
    #Tensorflow is channels-last
    if colorScheme=='gray':
        valid_shape = (len(images), imgSize[0], imgSize[1], 1)
    elif colorScheme=='rgb':
        valid_shape = (len(images), imgSize[0], imgSize[1], 3)
    else:
        raise('Invalid color scheme')

    #Create HDF5 matrices
    hdf5_file = h5py.File('HDF5data/Validation_{}_{}imgs+{}_{}_{}.h5'.format(DATASET,
                                                                             len(images),
                                                                             imgSize[0],
                                                                             imgSize[1],
                                                                             valid_shape[3]), mode='w')
    hdf5_file.create_dataset('valid_imgs', valid_shape, np.float32)
    hdf5_file.create_dataset('valid_labels', (len(images),2), np.float32)

    #Store Images
    for n, image in enumerate(images):
        if 'all.bmp' in image:
            print(n,'Valid-ALL')
            hdf5_file['valid_labels'][n] = ALL_LABEL
        if 'hem.bmp' in image:
            print(n,'Valid-HEM')
            hdf5_file['valid_labels'][n] = HEM_LABEL
        imgPath = '{}/{}'.format(VALIDATION_DATA_PATH, image)
        hdf5_file['valid_imgs'][n] = processImg(imgPath,
                                                subtractMean=subtractMean,
                                                divideStdDev=divideStdDev,
                                                colorScheme=colorScheme,
                                                imgSize=imgSize)
    #Close file
    hdf5_file.close()

def createHDF5_train(noOfImages, subtractMean=True, divideStdDev=True, colorScheme='rgb', imgSize=(450,450)):
    #Read Images
    images = os.listdir(TRAIN_DATA_PATH)
    np.random.shuffle(images)
    
    #Drop images until desired size
    droppedALL = droppedHEM = 0
    dropQtd = np.ceil((len(images) - noOfImages)/2)
    while len(images)>noOfImages:
        if ('hem.bmp' in images[0]) and (droppedHEM<dropQtd):
            droppedHEM+=1
            images.pop(0)
        if ('all.bmp' in images[0]) and (droppedALL<dropQtd):
            droppedALL+=1
            images.pop(0)
        np.random.shuffle(images)
        print('ALL: {}, HEM: {}'.format(droppedALL, droppedHEM))
    
    #Tensorflow is channels-last
    if colorScheme=='gray':
        train_shape = (len(images), imgSize[0], imgSize[1], 1)
    elif colorScheme=='rgb':
        train_shape = (len(images), imgSize[0], imgSize[1], 3)
    else:
        raise('Invalid color scheme')
    
    #Create HDF5 matrices
    hdf5_file = h5py.File('HDF5data/Train_{}_{}imgs+{}_{}_{}.h5'.format(DATASET,
                                                                        len(images),
                                                                        imgSize[0],
                                                                        imgSize[1],
                                                                        train_shape[3]), mode='w')
    hdf5_file.create_dataset('train_imgs', train_shape, np.float32)
    hdf5_file.create_dataset('train_labels', (len(images),2), np.float32)

    #Store Images
    for n, image in enumerate(images):
        if 'all.bmp' in image:
            print(n,'train-ALL')
            hdf5_file['train_labels'][n] = ALL_LABEL
        if 'hem.bmp' in image:
            print(n,'train-HEM')    
            hdf5_file['train_labels'][n] = HEM_LABEL
        imgPath = '{}/{}'.format(TRAIN_DATA_PATH, image)
        hdf5_file['train_imgs'][n] = processImg(imgPath,
                                                subtractMean=subtractMean,
                                                divideStdDev=divideStdDev,
                                                colorScheme=colorScheme,
                                                imgSize=imgSize)
    #Close file
    hdf5_file.close()

if __name__ == '__main__':

    import multiprocessing
    def createValidationSet():
        createHDF5_validation(noOfImages=5000, subtractMean=True, divideStdDev=False, colorScheme='rgb', imgSize=(250,250))
    
    def createTrainSet():
        createHDF5_train(noOfImages=20000, subtractMean=True, divideStdDev=False, colorScheme='rgb', imgSize=(250,250))
    #Spawn Process
    pTrain = multiprocessing.Process(name='HDF5_Train', target=createTrainSet)
    pValid = multiprocessing.Process(name='HDF5_Validation',target=createValidationSet)
    #pTrain.start()
    pValid.start()
    #pTrain.join()
    pValid.join()

    print("\nEnd Script!\n{}\n".format('#'*50))
