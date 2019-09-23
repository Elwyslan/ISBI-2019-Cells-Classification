import cv2
import numpy as np
import os
import shutil
from A_rawDataHandler import getCellsImgPath, getPatientCellsPath, getPatientsIDs, getPatientCellsPath, getIdsALLPatients, getIdsHEMPatients
from pathlib import Path
import multiprocessing

def removeBackground(image):
    if len(image.shape)==3:
        mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _,mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        _, contours, _ = cv2.findContours(mask, 1, 2)
        c = max(contours, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        return image[y:y+h, x:x+w, :]
    elif len(image.shape)==2:
        _,mask = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
        _, contours, _ = cv2.findContours(mask, 1, 2)
        c = max(contours, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        return image[y:y+h, x:x+w]

"""
Shear:
A shear parallel to the x axis results in:
    x' = x + shearV*y
    y' = y
A shear parallel to the y axis results in:
    x' = x
    y' = y + shearV*x
"""
def shearImage(image, s=(0.1, 0.35)):
    shearV = round(np.random.uniform(s[0], s[1]), 2)
    shearMatrix_X = np.array([[1.0, shearV, 0.0],
                              [0.0, 1.0, 0.0],
                              [0.0, 0.0, 1.0]])
    shearV = round(np.random.uniform(s[0], s[1]), 2)
    shearMatrix_Y = np.array([[1.0, 0.0, 0.0],
                              [shearV, 1.0, 0.0],
                              [0.0, 0.0, 1.0]])
    img = cv2.copyMakeBorder(image,225,225,225,225,cv2.BORDER_CONSTANT,value=[0,0,0])
    height, width, _ = img.shape
    shearAxis = np.random.choice([-1, 0, 1])
    if shearAxis==-1:
        img = cv2.warpPerspective(img, shearMatrix_X, (height,width))
    elif shearV==1:
        img = cv2.warpPerspective(img, shearMatrix_Y, (height,width))
    else:
        img = cv2.warpPerspective(img, shearMatrix_X, (height,width))
        img = cv2.warpPerspective(img, shearMatrix_Y, (height,width))
    img = removeBackground(img)
    w, h, _ = img.shape
    img = cv2.copyMakeBorder(img, ((450-w)//2)+1, ((450-w)//2)+1,
                                  ((450-h)//2)+1, ((450-h)//2)+1,
                                  cv2.BORDER_CONSTANT,value=[0,0,0])
    img = img[0:450, 0:450, :]
    return img

"""
Salt and Pepper: Salt and Pepper noise refers to addition of white and black dots in the image. 
"""
def saltPepperNoise(image, salt_vs_pepper=0.2, amount = 0.004):
    height, width, _ = image.shape
    #Create Salt and Pepper masks
    saltMask = np.random.uniform(0.0, 1.0, (height, width))
    pepperMask = np.random.uniform(0.0, 1.0, (height, width))
    saltMask[saltMask<=salt_vs_pepper] = 255.0
    pepperMask[pepperMask>salt_vs_pepper] = 255.0
    #pepperMask = pepperMask.astype(np.uint8)
    #saltMask = saltMask.astype(np.uint8)
    _, saltMask = cv2.threshold(saltMask, 127, 255, cv2.THRESH_BINARY)
    _, pepperMask = cv2.threshold(pepperMask, 127, 255, cv2.THRESH_BINARY)
    #Drop 'amount' pixels from saltMask
    toDrop = np.argwhere(saltMask>0)
    dropQtd = np.ceil(len(toDrop)*(1-amount)).astype(np.int32)
    np.random.shuffle(toDrop)
    for i in range(dropQtd):
        saltMask[toDrop[i][0],toDrop[i][1]] = 0
    #Drop 'amount' pixels from pepperMask
    toDrop = np.argwhere(pepperMask>0)
    dropQtd = np.ceil(len(toDrop)*(1-amount)).astype(np.int32)
    np.random.shuffle(toDrop)
    for i in range(dropQtd):
        pepperMask[toDrop[i][0],toDrop[i][1]] = 0
    #Apply pepperMasks on Image
    image = image.astype(np.int32)
    image[:,:,0] = np.add(image[:,:,0],pepperMask)
    image[:,:,1] = np.add(image[:,:,1],pepperMask)
    image[:,:,2] = np.add(image[:,:,2],pepperMask)
    #Apply saltMask on Image
    image[:,:,0] = np.subtract(image[:,:,0],saltMask)
    image[:,:,1] = np.subtract(image[:,:,1],saltMask)
    image[:,:,2] = np.subtract(image[:,:,2],saltMask)
    return np.clip(image, 0, 255).astype(np.uint8)


"""
Generate an Augmented Image base on srcImg
"""
def genAugmentedImage(srcImg):
    #Flip
    flipNum = int(np.random.choice([-1, 0, 1]))
    augm = cv2.flip(srcImg, flipNum)#horizontalANDVertical OR Horizontal OR vertical Flips
    #Rotation
    theta = np.random.randint(20, 340)
    height, width, _ = augm.shape
    rotationMatrix = cv2.getRotationMatrix2D((width/2,height/2), theta ,1)
    augm = cv2.warpAffine(augm, rotationMatrix,(width, height))
    #Aditional Augmentation
    rnd = np.random.uniform(0.0, 1.0)
    #25% Only Flip+Rotation
    if rnd < 0.25:
        return augm
    #25% Flip+Rotation+Shear
    if rnd >= 0.25 and rnd < 0.50:
        augm = shearImage(augm, s=(0.1, 0.35))
    #25% Flip+Rotation+PepperSalt
    if rnd >= 0.50 and rnd < 0.75:
        augm = saltPepperNoise(augm, salt_vs_pepper=0.50, amount = 0.02)
    #25% Flip+GaussianBlur
    if rnd >= 0.75:
        augm = cv2.GaussianBlur(augm,(5,5),0)
    return augm

def createDatasets(trainSize, validationSize):
    rndDiv_train = Path('data/rndDiv_train/')
    rndDiv_valid = Path('data/rndDiv_valid/')
    rndDiv_test = Path('data/rndDiv_test/')
    
    patLvDiv_train = Path('data/patLvDiv_train/')
    patLvDiv_valid = Path('data/patLvDiv_valid/')
    patLvDiv_test = Path('data/patLvDiv_test/')

    augm_patLvDiv_train = Path('data/augm_patLvDiv_train/')
    augm_patLvDiv_valid = Path('data/augm_patLvDiv_valid/')
    augm_rndDiv_train = Path('data/augm_rndDiv_train/')
    augm_rndDiv_valid = Path('data/augm_rndDiv_valid/')

    #Create folder 'data', if already exist, delete it
    if not os.path.exists(Path('data/')):
        os.mkdir(Path('data/'))
    else:
        shutil.rmtree(Path('data/'))
        os.mkdir(Path('data/'))

    #Create 10 Folders to store 10 datasets
    os.mkdir(rndDiv_train)
    os.mkdir(rndDiv_valid)
    os.mkdir(rndDiv_test)
    os.mkdir(patLvDiv_train)
    os.mkdir(patLvDiv_valid)
    os.mkdir(patLvDiv_test)
    os.mkdir(augm_patLvDiv_train)
    os.mkdir(augm_patLvDiv_valid)
    os.mkdir(augm_rndDiv_train)
    os.mkdir(augm_rndDiv_valid)
    
    """
    #*************************************************************************#
    SPLIT DATASET BY PATIENT LEVEL - DIVISAO POR PACIENTE
    #*************************************************************************#
    """
    ALLtrainPat = []
    ALLvalidPat = []
    ALLtestPat = []
    #Retrieve patient IDs diagnosed with ALL
    ALLpatientsIDs = getIdsALLPatients()

    #60% of ALL patientes for TRAIN
    maxPatTrain = int(np.ceil(len(ALLpatientsIDs)*0.6))

    #30% of ALL patientes for VALIDATION
    maxPatValid = int(np.floor(len(ALLpatientsIDs)*0.3))

    #Randomly populate 'ALLtrainPat' and 'ALLvalidPat' with TRAIN and VALIDATION patients
    while len(ALLpatientsIDs) > 0:
        np.random.shuffle(ALLpatientsIDs)
        if len(ALLtrainPat) < maxPatTrain:
            ALLtrainPat.append(ALLpatientsIDs.pop())
            continue
        if len(ALLvalidPat) < maxPatValid:
            ALLvalidPat.append(ALLpatientsIDs.pop())
            continue
        #Populate remaing patients for TEST
        ALLtestPat.append(ALLpatientsIDs.pop())
    
    HEMtrainPat = []
    HEMvalidPat = []
    HEMtestPat = []
    #Retrieve healthy patient IDs
    HEMpatientsIDs = getIdsHEMPatients()

    #60% of ALL patientes for TRAIN
    maxPatTrain = int(np.ceil(len(HEMpatientsIDs)*0.6))

    #30% of ALL patientes for Validation
    maxPatValid = int(np.floor(len(HEMpatientsIDs)*0.3))
    
    #Randomly populate 'HEMtrainPat' and 'HEMvalidPat' with TRAIN and VALIDATION patients
    while len(HEMpatientsIDs) > 0:
        np.random.shuffle(HEMpatientsIDs)
        if len(HEMtrainPat) < maxPatTrain:
            HEMtrainPat.append(HEMpatientsIDs.pop())
            continue
        if len(HEMvalidPat) < maxPatValid:
            HEMvalidPat.append(HEMpatientsIDs.pop())
            continue
        #Populate remaing patients for TEST
        HEMtestPat.append(HEMpatientsIDs.pop())

    #Define TRAIN, VALIDATION and TEST patients IDs
    trainPat = HEMtrainPat+ALLtrainPat
    validPat = HEMvalidPat+ALLvalidPat
    testPat = HEMtestPat+ALLtestPat
    print(f'Patients ALL in Train: {len(ALLtrainPat)}, Patients HEM in Train: {len(HEMtrainPat)}')
    print(f'Patients ALL in Validation: {len(ALLvalidPat)}, Patients HEM in Validation: {len(HEMvalidPat)}')
    print(f'Patients ALL in Test: {len(ALLtestPat)}, Patients HEM in Test: {len(HEMtestPat)}')

    #For each Train patient...
    for pId in trainPat:
        #Get patient cells
        pCells = getPatientCellsPath(pId)
        #Copy each patient cells into folder 'patLvDiv_train'
        for cellpath in pCells:
            shutil.copy2(cellpath, patLvDiv_train)
            print(f'Copy {cellpath} TO {patLvDiv_train}/{cellpath.name}')

    #For each Validation patient...
    for pId in validPat:
        #Get patient cells
        pCells = getPatientCellsPath(pId)
        #Copy each patient cells into folder 'patLvDiv_valid'
        for cellpath in pCells:
            shutil.copy2(cellpath, patLvDiv_valid)
            print(f'Copy {cellpath} TO {patLvDiv_valid}/{cellpath.name}')

    #For each Test patient...
    for pId in testPat:
        #Get patient cells
        pCells = getPatientCellsPath(pId)
        #Copy each patient cells into folder 'patLvDiv_test'
        for cellpath in pCells:
            shutil.copy2(cellpath, patLvDiv_test)
            print(f'Copy {cellpath} TO {patLvDiv_test}/{cellpath.name}')

    
    """
    #*************************************************************************#
    RAMDOM SPLIT DATASET - DIVISAO ALEATORIA
    #*************************************************************************#
    """
    ALLtrainImgs = []
    ALLvalidImgs = []
    ALLtestImgs = []
    #Retrieve every ALL cell in dataset
    ALLcells = getCellsImgPath('ALL')
    
    #60% of ALL cells for TRAIN
    maxTrainSize = int(np.ceil(len(ALLcells)*0.6))

    #30% of ALL cells for VALIDATION
    maxValidSize = int(np.floor(len(ALLcells)*0.3))

    #Randomly populate 'ALLtrainImgs' and 'ALLvalidImgs' with TRAIN and VALIDATION ALL cells
    while len(ALLcells) > 0:
        np.random.shuffle(ALLcells)
        if len(ALLtrainImgs) < maxTrainSize:
            ALLtrainImgs.append(ALLcells.pop())
            continue
        if len(ALLvalidImgs) < maxValidSize:
            ALLvalidImgs.append(ALLcells.pop())
            continue
        #Populate remaing ALL cells for TEST
        ALLtestImgs.append(ALLcells.pop())

    HEMtrainImgs = []
    HEMvalidImgs = []
    HEMtestImgs = []
    #Retrieve every HEM cell in dataset
    HEMcells = getCellsImgPath('HEM')

    #60% of HEM cells for TRAIN
    maxTrainSize = int(np.ceil(len(HEMcells)*0.6))

    #30% of HEM cells for VALIDATION
    maxValidSize = int(np.floor(len(HEMcells)*0.3))

    #Randomly populate 'HEMtrainImgs' and 'HEMvalidImgs' with TRAIN and VALIDATION HEM cells
    while len(HEMcells) > 0:
        np.random.shuffle(HEMcells)
        if len(HEMtrainImgs) < maxTrainSize:
            HEMtrainImgs.append(HEMcells.pop())
            continue
        if len(HEMvalidImgs) < maxValidSize:
            HEMvalidImgs.append(HEMcells.pop())
            continue
        #Populate remaing HEM cells for TEST
        HEMtestImgs.append(HEMcells.pop())

    #Define TRAIN, VALIDATION and TEST
    trainImgs = ALLtrainImgs + HEMtrainImgs
    validImgs = ALLvalidImgs + HEMvalidImgs
    testImgs = ALLtestImgs + HEMtestImgs
    print(f'ALL cells in Train: {len(ALLtrainImgs)}, HEM cells in Train: {len(HEMtrainImgs)}')
    print(f'ALL cells in Validation: {len(ALLvalidImgs)}, HEM cells in Validation: {len(HEMvalidImgs)}')
    print(f'ALL cells in Test: {len(ALLtestImgs)}, HEM cells in Test: {len(HEMtestImgs)}')

    #For each Train image...
    for cellpath in trainImgs:
        #Copy the image into folder 'rndDiv_train'
        shutil.copy2(cellpath, rndDiv_train)
        print(f'Copy {cellpath} TO {rndDiv_train}/{cellpath.name}')

    #For each Validation image...
    for cellpath in validImgs:
        #Copy the image into folder 'rndDiv_valid'
        shutil.copy2(cellpath, rndDiv_valid)
        print(f'Copy {cellpath} TO {rndDiv_valid}/{cellpath.name}')

    #For each TEST image...
    for cellpath in testImgs:
        #Copy the image into folder 'rndDiv_test'
        shutil.copy2(cellpath, rndDiv_test)
        print(f'Copy {cellpath} TO {rndDiv_test}/{cellpath.name}')

    #******************************************************************#
    #******************************************************************#
    #******************************************************************#

    #Thread to create 'Augm_rndDiv_train'
    def createAugm_rndDiv_train():
        countALL = 0 #Count how many ALL cells has in 'augm_rndDiv_train'
        countHEM = 0 #Count how many HEM cells has in 'augm_rndDiv_train'

        #For each cell in 'rndDiv_train' folder
        for cellpath in os.listdir(rndDiv_train):
            if cellpath.split('_')[-1]=='all.bmp':
                countALL += 1
            elif cellpath.split('_')[-1]=='hem.bmp':
                countHEM += 1
            else:
                continue
            #Copy cell into 'augm_rndDiv_train' folder
            shutil.copy2(rndDiv_train/cellpath, augm_rndDiv_train)
            print(f'Copy {rndDiv_train/cellpath} TO {augm_rndDiv_train}/{cellpath}')
        
        #Read all cells in 'rndDiv_train' folder
        srcTrain = os.listdir(rndDiv_train)

        #Until 'augm_rndDiv_train' folder didn't reach desired size...
        while len(os.listdir(augm_rndDiv_train)) < trainSize:
            #Randomly choose a cell from 'rndDiv_train' folder
            rndChoice = np.random.choice(srcTrain)
            #Logic to keep a balanced dataset (number of ALL cells equal to number of HEM cells)
            if rndChoice.split('_')[-1]=='all.bmp' and countALL<trainSize//2:
                countALL += 1
            elif rndChoice.split('_')[-1]=='hem.bmp' and countHEM<trainSize//2:
                countHEM += 1
            else:
                print('\nERROR in augm_rndDiv_train')
                print(f'Choice:{rndChoice}, countALL:{countALL}, countHEM:{countHEM}, trainSize//2:{trainSize//2}\n')
                continue
            #Create an augmented cell based on randomly choose from 'rndDiv_train' folder
            img = rndDiv_train / rndChoice
            img = genAugmentedImage(cv2.imread(str(img)))
            #Save the augmented image into folder 'augm_rndDiv_train'
            savePath = augm_rndDiv_train / f'AugmentedImg_{np.random.randint(1001, 9999)}_{rndChoice}'
            if not os.path.isfile(savePath):
                cv2.imwrite(str(savePath), img)
                print(f'Created {savePath}')
            else:
                #Logic to keep a balanced dataset (number of ALL cells equal to number of HEM cells)
                if rndChoice.split('_')[-1]=='all.bmp':
                    countALL -= 1
                elif rndChoice.split('_')[-1]=='hem.bmp':
                    countHEM -= 1

    #Thread to create 'Augm_rndDiv_valid'
    def createAugm_rndDiv_valid():
        countALL = 0 #Count how many ALL cells has in 'augm_rndDiv_valid'
        countHEM = 0 #Count how many HEM cells has in 'augm_rndDiv_valid'

        #For each cell in 'rndDiv_valid' folder
        for cellpath in os.listdir(rndDiv_valid):
            if cellpath.split('_')[-1]=='all.bmp':
                countALL += 1
            elif cellpath.split('_')[-1]=='hem.bmp':
                countHEM += 1
            else:
                continue
            #Copy cell into 'augm_rndDiv_valid' folder
            shutil.copy2(rndDiv_valid/cellpath, augm_rndDiv_valid)
            print(f'Copy {rndDiv_valid/cellpath} TO {augm_rndDiv_valid}/{cellpath}')
        
        #Read all cells in 'rndDiv_valid' folder
        srcValid = os.listdir(rndDiv_valid)

        #Until 'augm_rndDiv_valid' folder didn't reach desired size...
        while len(os.listdir(augm_rndDiv_valid)) < validationSize:
            #Randomly choose a cell from 'rndDiv_valid' folder
            rndChoice = np.random.choice(srcValid)
            #Logic to keep a balanced dataset (number of ALL cells equal to number of HEM cells)
            if rndChoice.split('_')[-1]=='all.bmp' and countALL<validationSize//2:
                countALL += 1
            elif  rndChoice.split('_')[-1]=='hem.bmp' and countHEM<validationSize//2:
                countHEM += 1
            else:
                print('\nERROR in augm_rndDiv_valid')
                print(f'Choice:{rndChoice}, countALL:{countALL}, countHEM:{countHEM}, validationSize//2:{validationSize//2}\n')
                continue
            #Create an augmented cell based on randomly choose from 'rndDiv_valid' folder
            img = rndDiv_valid / rndChoice
            img = genAugmentedImage(cv2.imread(str(img)))
            #Save the augmented image into folder 'augm_rndDiv_valid'
            savePath = augm_rndDiv_valid / f'AugmentedImg_{np.random.randint(1001, 9999)}_{rndChoice}'
            if not os.path.isfile(savePath):
                cv2.imwrite(str(savePath), img)
                print(f'Created {savePath}')
            else:
                #Logic to keep a balanced dataset (number of ALL cells equal to number of HEM cells)
                if rndChoice.split('_')[-1]=='all.bmp':
                    countALL -= 1
                elif rndChoice.split('_')[-1]=='hem.bmp':
                    countHEM -= 1

    #Thread to create 'Augm_patLvDiv_train'
    def createAugm_patLvDiv_train():
        countALL = 0 #Count how many ALL cells has in 'augm_patLvDiv_train'
        countHEM = 0 #Count how many HEM cells has in 'augm_patLvDiv_train'

        #For each cell in 'patLvDiv_train' folder
        for cellpath in os.listdir(patLvDiv_train):
            if cellpath.split('_')[-1]=='all.bmp':
                countALL += 1
            elif cellpath.split('_')[-1]=='hem.bmp':
                countHEM += 1
            else:
                continue
            #Copy cell into 'augm_patLvDiv_train' folder
            shutil.copy2(patLvDiv_train/cellpath, augm_patLvDiv_train)
            print(f'Copy {patLvDiv_train/cellpath} TO {augm_patLvDiv_train}/{cellpath}')

        #Read all cells in 'patLvDiv_train' folder
        srcTrain = os.listdir(patLvDiv_train)

        #Until 'augm_patLvDiv_train' folder didn't reach desired size...
        while len(os.listdir(augm_patLvDiv_train)) < trainSize:
            #Randomly choose a cell from 'patLvDiv_train' folder
            rndChoice = np.random.choice(srcTrain)
            #Logic to keep a balanced dataset (number of ALL cells equal to number of HEM cells)
            if rndChoice.split('_')[-1]=='all.bmp' and countALL<trainSize//2:
                countALL += 1
            elif rndChoice.split('_')[-1]=='hem.bmp' and countHEM<trainSize//2:
                countHEM += 1
            else:
                print('\nERROR in augm_patLvDiv_train')
                print(f'Choice:{rndChoice}, countALL:{countALL}, countHEM:{countHEM}, trainSize//2:{trainSize//2}\n')
                continue
            #Create an augmented cell based on randomly choose from 'patLvDiv_train' folder
            img = patLvDiv_train / rndChoice
            img = genAugmentedImage(cv2.imread(str(img)))
            #Save the augmented image into folder 'augm_patLvDiv_train'
            savePath = augm_patLvDiv_train / f'AugmentedImg_{np.random.randint(1001, 9999)}_{rndChoice}'
            if not os.path.isfile(savePath):
                cv2.imwrite(str(savePath), img)
                print(f'Created {savePath}')
            else:
                #Logic to keep a balanced dataset (number of ALL cells equal to number of HEM cells)
                if rndChoice.split('_')[-1]=='all.bmp':
                    countALL -= 1
                elif rndChoice.split('_')[-1]=='hem.bmp':
                    countHEM -= 1

    #Thread to create 'Augm_patLvDiv_valid'
    def createAugm_patLvDiv_valid():
        countALL = 0 #Count how many ALL cells has in 'augm_patLvDiv_train'
        countHEM = 0 #Count how many HEM cells has in 'augm_patLvDiv_train'

        #For each cell in 'patLvDiv_valid' folder
        for cellpath in os.listdir(patLvDiv_valid):
            if cellpath.split('_')[-1]=='all.bmp':
                countALL += 1
            elif cellpath.split('_')[-1]=='hem.bmp':
                countHEM += 1
            else:
                continue
            #Copy cell into 'augm_patLvDiv_valid' folder
            shutil.copy2(patLvDiv_valid/cellpath, augm_patLvDiv_valid)
            print(f'Copy {patLvDiv_valid/cellpath} TO {augm_patLvDiv_valid}/{cellpath}')

        #Read all cells in 'patLvDiv_valid' folder
        srcValid = os.listdir(patLvDiv_valid)

        #Until 'augm_patLvDiv_valid' folder didn't reach desired size...
        while len(os.listdir(augm_patLvDiv_valid)) < validationSize:
            #Randomly choose a cell from 'patLvDiv_valid' folder
            rndChoice = np.random.choice(srcValid)
            #Logic to keep a balanced dataset (number of ALL cells equal to number of HEM cells)
            if rndChoice.split('_')[-1]=='all.bmp' and countALL<validationSize//2:
                countALL += 1
            elif rndChoice.split('_')[-1]=='hem.bmp' and countHEM<validationSize//2:
                countHEM += 1
            else:
                print('\nERROR in augm_patLvDiv_valid')
                print(f'Choice:{rndChoice}, countALL:{countALL}, countHEM:{countHEM}, validationSize//2:{validationSize//2}\n')
                continue
            #Create an augmented cell based on randomly choose from 'patLvDiv_valid' folder
            img = patLvDiv_valid / rndChoice
            img = genAugmentedImage(cv2.imread(str(img)))
            #Save the augmented image into folder 'augm_patLvDiv_valid'
            savePath = augm_patLvDiv_valid / f'AugmentedImg_{np.random.randint(1001, 9999)}_{rndChoice}'
            if not os.path.isfile(savePath):
                cv2.imwrite(str(savePath), img)
                print(f'Created {savePath}')
            else:
                #Logic to keep a balanced dataset (number of ALL cells equal to number of HEM cells)
                if rndChoice.split('_')[-1]=='all.bmp':
                    countALL -= 1
                elif rndChoice.split('_')[-1]=='hem.bmp':
                    countHEM -= 1

    #Create Augmented Datasets
    pTrain1 = multiprocessing.Process(name='Train1 Augm', target=createAugm_rndDiv_train)
    pValid1 = multiprocessing.Process(name='Validation1 Augm', target=createAugm_rndDiv_valid)
    pTrain2 = multiprocessing.Process(name='Train2 Augm', target=createAugm_patLvDiv_train)
    pValid2 = multiprocessing.Process(name='Validation2 Augm', target=createAugm_patLvDiv_valid)
    pTrain1.start()
    pValid1.start()
    pTrain2.start()
    pValid2.start()

    pTrain1.join()
    pValid1.join()
    pTrain2.join()
    pValid2.join()


def folderData(folderPath):
    folderName = str(folderPath).split('/')[1]
    print(f'Folder Name: {folderName}')

    countALL = countHEM = 0
    patientsIDs = []
    for cell in os.listdir(folderPath):
        if 'all.bmp' in cell:
            countALL += 1
        elif 'hem.bmp' in cell:
            countHEM += 1

        if cell.split('_')[0] == 'AugmentedImg':
            patientID = cell.split('_')[3]
        else:
            patientID = cell.split('_')[1]
        patientsIDs.append(patientID)
    patientsIDs = list(set(patientsIDs))
    patientsIDs.sort()
    print(f'Number of HEM cells: {countHEM}')
    print(f'Number of ALL cells: {countALL}')
    print(f'Patients IDs: {patientsIDs}')
    print('\n')
    return countALL, countHEM, patientsIDs 


if __name__ == '__main__':
    #createDatasets(trainSize=20000, validationSize=5000)

    createDatasets(trainSize=200, validationSize=50) #Sample Data
    print(f"\nEnd Script!\n{'#'*50}")

