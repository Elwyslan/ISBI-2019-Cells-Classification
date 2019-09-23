from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('ggplot')

def formatNumber(num):
    return list(map(lambda n:f'{n:.2f}'.replace('.',','), [num]))[0]

def printValidationTable():
    validationData = Path('results/ConvNet_ValidPerformance.csv')
    validationData = pd.read_csv(validationData, index_col=0)

    xception_acc_augmRndDiv, xception_acc_augmpatLvDiv = 0, 0
    vgg16_acc_augmRndDiv, vgg16_acc_augmpatLvDiv = 0, 0
    vgg19_acc_augmRndDiv, vgg19_acc_augmpatLvDiv = 0, 0
    resnet50_acc_augmRndDiv, resnet50_acc_augmpatLvDiv = 0, 0
    inceptionV3_acc_augmRndDiv, inceptionV3_acc_augmpatLvDiv = 0, 0
    for n in range(validationData.shape[0]):
        data = validationData.iloc[n].values.tolist()
        if 'xception' in data and 'RndDiv' in data:
            xception_acc_augmRndDiv = data[2]
        if 'xception' in data and 'PatDiv' in data:
            xception_acc_augmpatLvDiv = data[2]
        ##################################################
        if 'vgg16' in data and 'RndDiv' in data:
            vgg16_acc_augmRndDiv = data[2]
        if 'vgg16' in data and 'PatDiv' in data:
            vgg16_acc_augmpatLvDiv = data[2]
        ##################################################
        if 'vgg19' in data and 'RndDiv' in data:
            vgg19_acc_augmRndDiv = data[2]
        if 'vgg19' in data and 'PatDiv' in data:
            vgg19_acc_augmpatLvDiv = data[2]
        ##################################################
        if 'resNet50' in data and 'RndDiv' in data:
            resnet50_acc_augmRndDiv = data[2]
        if 'resNet50' in data and 'PatDiv' in data:
            resnet50_acc_augmpatLvDiv = data[2]
        ##################################################
        if 'inceptionV3' in data and 'RndDiv' in data:
            inceptionV3_acc_augmRndDiv = data[2]
        if 'inceptionV3' in data and 'PatDiv' in data:
            inceptionV3_acc_augmpatLvDiv = data[2]

    #print(xception_acc_augmRndDiv, xception_acc_augmpatLvDiv)
    #print(vgg16_acc_augmRndDiv, vgg16_acc_augmpatLvDiv)
    #print(vgg19_acc_augmRndDiv, vgg19_acc_augmpatLvDiv)
    #print(resnet50_acc_augmRndDiv, resnet50_acc_augmpatLvDiv)
    #print(inceptionV3_acc_augmRndDiv, inceptionV3_acc_augmpatLvDiv)

    lines = []
    lines.append('\\begin{tabular}{c|c|c}')
    lines.append('Arquitetura & \\makecell{Acurácia \\ (c/ imagens da Divisão por Paciente)} & \\makecell{Acurácia \\ (c/ imagens da Divisão Aleatória)} \\\\')
    lines.append('\\hline')
    lines.append(f'Xception    & {formatNumber(xception_acc_augmpatLvDiv)}\\% & {formatNumber(xception_acc_augmRndDiv)}\\% \\\\')
    lines.append(f'Vgg16       & {formatNumber(vgg16_acc_augmpatLvDiv)}\\% & {formatNumber(vgg16_acc_augmRndDiv)}\\% \\\\')
    lines.append(f'Vgg19       & {formatNumber(vgg19_acc_augmpatLvDiv)}\\% & {formatNumber(vgg19_acc_augmRndDiv)}\\% \\\\')
    lines.append(f'ResNet50    & {formatNumber(resnet50_acc_augmpatLvDiv)}\\% & {formatNumber(resnet50_acc_augmRndDiv)}\\% \\\\')
    lines.append(f'InceptionV3 & {formatNumber(inceptionV3_acc_augmpatLvDiv)}\\% & {formatNumber(inceptionV3_acc_augmRndDiv)}\\% \\\\ ')
    lines.append('\\hline')
    lines.append('\\end{tabular}')
    print('\n'.join(lines))


def plotTrainData():
    VGG16_AugmPatLvDiv = Path('results/VGG16_AugmPatLvDiv_TrainData_400-Epochs.csv')
    VGG16_AugmRndDiv = Path('results/VGG16_AugmRndDiv_TrainData_400-Epochs.csv')
    VGG19_AugmPatLvDiv = Path('results/VGG19_AugmPatLvDiv_TrainData_400-Epochs.csv')
    VGG19_AugmRndDiv = Path('results/VGG19_AugmRndDiv_TrainData_400-Epochs.csv')
    Xception_AugmPatLvDiv = Path('results/Xception_AugmPatLvDiv_TrainData_400-Epochs.csv')
    Xception_AugmRndDiv = Path('results/Xception_AugmRndDiv_TrainData_400-Epochs.csv')
    VGG16_AugmPatLvDiv = pd.read_csv(VGG16_AugmPatLvDiv, index_col=0)
    VGG16_AugmRndDiv = pd.read_csv(VGG16_AugmRndDiv, index_col=0)
    VGG19_AugmPatLvDiv = pd.read_csv(VGG19_AugmPatLvDiv, index_col=0)
    VGG19_AugmRndDiv = pd.read_csv(VGG19_AugmRndDiv, index_col=0)
    Xception_AugmPatLvDiv = pd.read_csv(Xception_AugmPatLvDiv, index_col=0)
    Xception_AugmRndDiv = pd.read_csv(Xception_AugmRndDiv, index_col=0)

    x = np.arange(0,400,1)

    ###############################################
    plt.figure(figsize=(10, 6))
    y = VGG16_AugmPatLvDiv['Training Accuracy'].values
    plt.plot(x, y, label='Acurácia de treinamento')
    y = VGG16_AugmPatLvDiv['Validation Accuracy'].values
    plt.plot(x, y, label='Acurácia de Validação')
    plt.legend(loc='lower right',prop={'size': 18, 'weight':'bold'})
    plt.ylabel('Acurácia (%)',fontsize=18)
    plt.xlabel('Épocas de Treinamento',fontsize=18)
    plt.tight_layout()
    plt.savefig(Path('VGG16_AugmPatLvDiv_TrainData.png'))
    plt.show()
    ###############################################
    plt.figure(figsize=(10, 6))
    y = VGG16_AugmRndDiv['Training Accuracy'].values
    plt.plot(x, y, label='Acurácia de treinamento')
    y = VGG16_AugmRndDiv['Validation Accuracy'].values
    plt.plot(x, y, label='Acurácia de Validação')
    plt.legend(loc='lower right',prop={'size': 18, 'weight':'bold'})
    plt.ylabel('Acurácia (%)',fontsize=18)
    plt.xlabel('Épocas de Treinamento',fontsize=18)
    plt.tight_layout()
    plt.savefig(Path('VGG16_AugmRndDiv_TrainData.png'))
    plt.show()
    ###############################################

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%$
    
    ###############################################
    plt.figure(figsize=(10, 6))
    y = VGG19_AugmPatLvDiv['Training Accuracy'].values
    plt.plot(x, y, label='Acurácia de treinamento')
    y = VGG19_AugmPatLvDiv['Validation Accuracy'].values
    plt.plot(x, y, label='Acurácia de Validação')
    plt.legend(loc='lower right',prop={'size': 18, 'weight':'bold'})
    plt.ylabel('Acurácia (%)',fontsize=18)
    plt.xlabel('Épocas de Treinamento',fontsize=18)
    plt.tight_layout()
    plt.savefig(Path('VGG19_AugmPatLvDiv_TrainData.png'))
    plt.show()
    ###############################################
    plt.figure(figsize=(10, 6))
    y = VGG19_AugmRndDiv['Training Accuracy'].values
    plt.plot(x, y, label='Acurácia de treinamento')
    y = VGG19_AugmRndDiv['Validation Accuracy'].values
    plt.plot(x, y, label='Acurácia de Validação')
    plt.legend(loc='lower right',prop={'size': 18, 'weight':'bold'})
    plt.ylabel('Acurácia (%)',fontsize=18)
    plt.xlabel('Épocas de Treinamento',fontsize=18)
    plt.tight_layout()
    plt.savefig(Path('VGG19_AugmRndDiv_TrainData.png'))
    plt.show()
    ###############################################

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%$
    
    ###############################################
    plt.figure(figsize=(10, 6))
    y = Xception_AugmPatLvDiv['Training Accuracy'].values
    plt.plot(x, y, label='Acurácia de treinamento')
    y = Xception_AugmPatLvDiv['Validation Accuracy'].values
    plt.plot(x, y, label='Acurácia de Validação')
    plt.legend(loc='lower right',prop={'size': 18, 'weight':'bold'})
    plt.ylabel('Acurácia (%)',fontsize=18)
    plt.xlabel('Épocas de Treinamento',fontsize=18)
    plt.tight_layout()
    plt.savefig(Path('Xception_AugmPatLvDiv_TrainData.png'))
    plt.show()
    ###############################################
    plt.figure(figsize=(10, 6))
    y = Xception_AugmRndDiv['Training Accuracy'].values
    plt.plot(x, y, label='Acurácia de treinamento')
    y = Xception_AugmRndDiv['Validation Accuracy'].values
    plt.plot(x, y, label='Acurácia de Validação')
    plt.legend(loc='lower right',prop={'size': 18, 'weight':'bold'})
    plt.ylabel('Acurácia (%)',fontsize=18)
    plt.xlabel('Épocas de Treinamento',fontsize=18)
    plt.tight_layout()
    plt.savefig(Path('Xception_AugmRndDiv_TrainData.png'))
    plt.show()
    ###############################################

def printXceptionTable():
    df = pd.read_csv(Path('results/ConvNet_TestPerformance.csv'))
    l00 = '\\begin{tabular}{l|c|ccccc}'
    l01 = '\\multicolumn{1}{c|}{Conj.Treino} & Conj.Teste & Acurácia & Precisão & Sensibilidade & Especificidade & \\textit{F1-score} \\\\'
    l02 = '\\cline{1-7}'

    vals = df.iloc[0].values[2:7].tolist()
    vals = list(map(lambda n:f'{n:.2f}'.replace('.',','), vals))
    l03 = f'Divisão por paciente (com augm.) & \\multirow{{1}}{{*}}{{\\textit{{patLvDiv\\_test}}}} & {vals[1]}\\% & {vals[2]}\\% & {vals[3]}\\% & {vals[4]}\\% & {vals[0]}\\% \\\\'

    l04 = '\\cline{1-7}'
    vals = df.iloc[1].values[2:7].tolist()
    vals = list(map(lambda n:f'{n:.2f}'.replace('.',','), vals))
    l05 = f'Divisão aleatória (com augm.)    & \\multirow{{1}}{{*}}{{\\textit{{rndDiv\\_test}}}}   & {vals[1]}\\% & {vals[2]}\\% & {vals[3]}\\% & {vals[4]}\\% & {vals[0]}\\% \\\\'

    l06 = '\\cline{1-7}'
    l07 = '\\end{tabular}'

    print('\n'.join([l00,l01,l02,l03,l04,l05,l06,l07]))

def printVGG16Table():
    df = pd.read_csv(Path('results/ConvNet_TestPerformance.csv'))
    l00 = '\\begin{tabular}{l|c|ccccc}'
    l01 = '\\multicolumn{1}{c|}{Conj.Treino} & Conj.Teste & Acurácia & Precisão & Sensibilidade & Especificidade & \\textit{F1-score} \\\\'
    l02 = '\\cline{1-7}'

    vals = df.iloc[2].values[2:7].tolist()
    vals = list(map(lambda n:f'{n:.2f}'.replace('.',','), vals))
    l03 = f'Divisão por paciente (com augm.) & \\multirow{{1}}{{*}}{{\\textit{{patLvDiv\\_test}}}} & {vals[1]}\\% & {vals[2]}\\% & {vals[3]}\\% & {vals[4]}\\% & {vals[0]}\\% \\\\'

    l04 = '\\cline{1-7}'
    vals = df.iloc[3].values[2:7].tolist()
    vals = list(map(lambda n:f'{n:.2f}'.replace('.',','), vals))
    l05 = f'Divisão aleatória (com augm.)    & \\multirow{{1}}{{*}}{{\\textit{{rndDiv\\_test}}}}   & {vals[1]}\\% & {vals[2]}\\% & {vals[3]}\\% & {vals[4]}\\% & {vals[0]}\\% \\\\'

    l06 = '\\cline{1-7}'
    l07 = '\\end{tabular}'

    print('\n'.join([l00,l01,l02,l03,l04,l05,l06,l07]))

def printVGG19Table():
    df = pd.read_csv(Path('results/ConvNet_TestPerformance.csv'))
    l00 = '\\begin{tabular}{l|c|ccccc}'
    l01 = '\\multicolumn{1}{c|}{Conj.Treino} & Conj.Teste & Acurácia & Precisão & Sensibilidade & Especificidade & \\textit{F1-score} \\\\'
    l02 = '\\cline{1-7}'

    vals = df.iloc[4].values[2:7].tolist()
    vals = list(map(lambda n:f'{n:.2f}'.replace('.',','), vals))
    l03 = f'Divisão por paciente (com augm.) & \\multirow{{1}}{{*}}{{\\textit{{patLvDiv\\_test}}}} & {vals[1]}\\% & {vals[2]}\\% & {vals[3]}\\% & {vals[4]}\\% & {vals[0]}\\% \\\\'

    l04 = '\\cline{1-7}'
    vals = df.iloc[5].values[2:7].tolist()
    vals = list(map(lambda n:f'{n:.2f}'.replace('.',','), vals))
    l05 = f'Divisão aleatória (com augm.)    & \\multirow{{1}}{{*}}{{\\textit{{rndDiv\\_test}}}}   & {vals[1]}\\% & {vals[2]}\\% & {vals[3]}\\% & {vals[4]}\\% & {vals[0]}\\% \\\\'

    l06 = '\\cline{1-7}'
    l07 = '\\end{tabular}'

    print('\n'.join([l00,l01,l02,l03,l04,l05,l06,l07]))

if __name__ == '__main__':    
    #printValidationTable()
    #plotTrainData()
    #printXceptionTable()
    #printVGG16Table()
    printVGG19Table()
    print(f"\nEnd Script!\n{'#'*50}")
