import sys
from pathlib import Path

trainPath = Path('train/')
trainPath_ALL0 = trainPath / Path('fold_0/all/')
trainPath_ALL1 = trainPath / Path('fold_1/all/')
trainPath_ALL2 = trainPath / Path('fold_2/all/')
trainPath_HEM0 = trainPath / Path('fold_0/hem/')
trainPath_HEM1 = trainPath / Path('fold_1/hem/')
trainPath_HEM2 = trainPath / Path('fold_2/hem/')


class Patient():
    def __init__(self, patientID):
        self.patientID = patientID
        self.listOfALLCells = []
        self.listOfHEMCells = []

    def addALLCell(self, imgPath):
        self.listOfALLCells.append(imgPath)

    def addHEMCell(self, imgPath):
        self.listOfHEMCells.append(imgPath)

    def ALLCellsCount(self):
        return len(self.listOfALLCells)

    def HEMCellsCount(self):
        return len(self.listOfHEMCells)


def totalCells(cellType):
    global patients
    total = 0
    for _, patient in patients.items():
        if cellType=='ALL':
            total += patient.ALLCellsCount()
        elif cellType=='HEM':
            total += patient.HEMCellsCount()
        else:
            raise Exception('Especify cell type HEM or ALL')

    return total

def getCellsImgPath(cellType):
    global patients
    ret = []
    for _, patient in patients.items():
        if cellType=='ALL':
            ret += patient.listOfALLCells
        elif cellType=='HEM':
            ret += patient.listOfHEMCells
        else:
            raise Exception('Especify cell type HEM or ALL')

    return ret

def getPatientCellsPath(patientID):
    global patients
    patient = patients[patientID]
    if patient.ALLCellsCount()==0:
        return patient.listOfHEMCells.copy()
    else:
        return patient.listOfALLCells.copy()

def iteratorALL(cellType):
    allCells = getCellsImgPath(cellType)
    while len(allCells)>0:
        yield allCells.pop()

def getPatientsIDs():
    global patients
    return list(patients.keys())

def getIdsALLPatients():
    global patients
    ret = []
    for pId, patient in patients.items():
        if patient.ALLCellsCount()>0:
            ret.append(pId)
    return ret

def getIdsHEMPatients():
    global patients
    ret = []
    for pId, patient in patients.items():
        if patient.HEMCellsCount()>0:
            ret.append(pId)
    return ret

patients = {}
allTrainImgs = trainPath.glob('*/*/*.bmp')# fold_[0,1,2]/[all,hem]/*bmp
for imgPath in allTrainImgs:
    patientID = imgPath.stem.split('_')[1]
    cellType = imgPath.stem.split('_')[-1]
    if patientID not in patients:
        newPatient = Patient(patientID)
        patients[patientID] = newPatient
    
    if cellType=='all':
        patients[patientID].addALLCell(imgPath)
    else:
        patients[patientID].addHEMCell(imgPath)

if __name__ == '__main__':
    for patientID, patient in patients.items():
        print(f"ID:{patientID}; ALL count:{patient.ALLCellsCount()}; HEM count:{patient.HEMCellsCount()} ")

    print(f"Total HEM:{totalCells(cellType='HEM')}")
    print(f"Total ALL:{totalCells(cellType='ALL')}")
    print(f"Total Patients:{len(patients.keys())}")

    patientsIDs = getPatientsIDs()
    trainALL = trainHEM = validALL = validHEM = 0
    for pId in patientsIDs:
        pCells = getPatientCellsPath(pId)
        trainSplit = pCells[:len(pCells)//2]
        validSplit = pCells[len(pCells)//2:]
        for cellPath in trainSplit:
            if cellPath.stem.split('_')[-1]=='all':
                trainALL+=1
            elif cellPath.stem.split('_')[-1]=='hem':
                trainHEM+=1

        for cellPath in validSplit:
            if cellPath.stem.split('_')[-1]=='all':
                validALL+=1
            elif cellPath.stem.split('_')[-1]=='hem':
                validHEM+=1
    print(f'trainALL = {trainALL}, trainHEM = {trainHEM}, validALL = {validALL}, validHEM={validHEM}')
    print(f"\nEnd Script!\n{'#'*50}")
