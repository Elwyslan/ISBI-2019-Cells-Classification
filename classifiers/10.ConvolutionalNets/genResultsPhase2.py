import os
import numpy as np
import pandas as pd
from pathlib import Path
import zipfile
from keras.models import load_model
from C_executeOnTest import processImg

VGG16_AugmPatDiv = Path('Best Models/VGG16_AugmPatDiv.h5')
VGG16_AugmRndDiv = Path('Best Models/VGG16_AugmRndDiv.h5')
VGG19_augmPatLvDiv = Path('Best Models/VGG19_augmPatLvDiv.h5')
VGG19_augmRndDiv = Path('Best Models/VGG19_augmRndDiv.h5')
xception_AugmPatDiv = Path('Best Models/xception_AugmPatDiv.h5')
xception_augmRndDiv = Path('Best Models/xception_augmRndDiv.h5')

PHASE2_DATA = Path('../../C-NMC_phase_2_data/')
SAVEFILE = Path('isbi_valid.predict')

if __name__ == '__main__':
    imgs = {}
    for img in os.listdir(PHASE2_DATA):
        imgs[int(img.split('.')[0])] = PHASE2_DATA / img

    model = load_model(str(xception_augmRndDiv))
    with open(SAVEFILE, 'w') as f:
        allCount=0
        hemCount=0
        for key in range(1,len(imgs)+1):
            img = processImg(imgs[key], subtractMean=True, divideStdDev=False, colorScheme='rgb', imgSize=(250,250))
            predict = model.predict(np.array([img],dtype=np.float32), verbose=1)
            if predict[0][0] > predict[0][1]:
                f.write('1\n')
                print(f'{imgs[key]} - ALL')
                allCount+=1
            else:
                f.write('0\n')
                print(f'{imgs[key]} - HEM')
                hemCount+=1
        print(f'ALL:{allCount} :: HEM:{hemCount}')
        f.flush()
    zipSubmission = zipfile.ZipFile('isbi_valid_xception_augmRndDiv.zip', 'w')
    zipSubmission.write(SAVEFILE, compress_type=zipfile.ZIP_DEFLATED)
    zipSubmission.close()
    os.remove(SAVEFILE)

    print(f"\nEnd Script!\n{'#'*50}")


"""
with open(SAVEFILE, 'w') as f:
        cols = list(getMorphologicalFeatures(imgs[1]).keys())
        df = pd.DataFrame(columns=cols)
        for key in range(1,len(imgs)+1):
            df = df.append(getMorphologicalFeatures(imgs[key]), ignore_index=True)
            print(f'{imgs[key]}')
        for col in df.columns:
            df[col] = (df[col] - df[col].mean()) / df[col].std() #mean=0, std=1
        x_valid = df.values

        allCount=0
        hemCount=0
        for key in range(len(x_valid)):
            #Do predictions
            predictions = clfs.morphological_CLF.predict(x_valid[key,:].reshape(1,-1))[0]
            #Check predictions - Test for ALL cells
            if predictions[0] > predictions[1]:#Model Predict ALL...
                f.write('1\n')
                print(f'{imgs[key+1]} - ALL')
                allCount+=1
            elif predictions[1] > predictions[0]:#Model Predict HEM...
                f.write('0\n')
                print(f'{imgs[key+1]} - HEM')
                hemCount+=1
        print(f'ALL:{allCount} :: HEM:{hemCount}')
        f.flush()
    zipSubmission = zipfile.ZipFile('isbi_valid_MorphologicalFeatures_TestGlobalNormalization.zip', 'w')
    zipSubmission.write(SAVEFILE, compress_type=zipfile.ZIP_DEFLATED)
    zipSubmission.close()
    os.remove(SAVEFILE)
"""