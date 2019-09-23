from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('ggplot')

def plotLSVM():
    LSVM_ValPerfAugmPatLvDivImages = Path('results/L-SVM_ValidPerformance_DCTData_AugmPatLvDivImages.csv')
    LSVM_ValPerfAugmRndDivImages = Path('results/L-SVM_ValidPerformance_DCTData_AugmRndDivImages.csv')
    LSVM_ValPerfPatLvDivImages = Path('results/L-SVM_ValidPerformance_DCTData_PatLvDivImages.csv')
    LSVM_ValPerfRndDivImages = Path('results/L-SVM_ValidPerformance_DCTData_RndDivImages.csv')
    LSVM_ValPerfAugmPatLvDivImages = pd.read_csv(LSVM_ValPerfAugmPatLvDivImages, index_col=0)
    LSVM_ValPerfAugmRndDivImages = pd.read_csv(LSVM_ValPerfAugmRndDivImages, index_col=0)
    LSVM_ValPerfPatLvDivImages = pd.read_csv(LSVM_ValPerfPatLvDivImages, index_col=0)
    LSVM_ValPerfRndDivImages = pd.read_csv(LSVM_ValPerfRndDivImages, index_col=0)

    x = LSVM_ValPerfAugmPatLvDivImages['Features'].values
    y0 = LSVM_ValPerfAugmPatLvDivImages['L-SVM_acc'].values
    y1 = LSVM_ValPerfAugmRndDivImages['L-SVM_acc'].values
    y2 = LSVM_ValPerfPatLvDivImages['L-SVM_acc'].values
    y3 = LSVM_ValPerfRndDivImages['L-SVM_acc'].values

    plt.figure(figsize=(10, 6))
    ###############################################
    plt.plot(x, y1, label='Divisão Aleatória com Data Augmentation')
    plt.annotate(f'{y1[np.argmax(y1)]:.2f}%'.replace('.',','),
                   xy=(x[np.argmax(y1)]-5,y1[np.argmax(y1)]-0.3), fontweight='bold', fontsize=12)
    plt.scatter(x[np.argmax(y1)], y1[np.argmax(y1)])
    ###############################################
    plt.plot(x, y3, label='Divisão Aleatória')
    plt.annotate(f'{y3[np.argmax(y3)]:.2f}%'.replace('.',','),
                   xy=(x[np.argmax(y3)],y3[np.argmax(y3)]+0.05), fontweight='bold', fontsize=12)
    plt.scatter(x[np.argmax(y3)], y3[np.argmax(y3)])
    ###############################################
    plt.plot(x, y0, label='Divisão por Paciente com Data Augmentation')
    plt.annotate(f'{y0[np.argmax(y0)]:.2f}%'.replace('.',','),
                 xy=(x[np.argmax(y0)],y0[np.argmax(y0)]+0.1), fontweight='bold', fontsize=12)
    plt.scatter(x[np.argmax(y0)], y0[np.argmax(y0)])
    ###############################################
    plt.plot(x, y2, label='Divisão por Paciente')
    plt.annotate(f'{y2[np.argmax(y2)]:.2f}%'.replace('.',','),
                   xy=(x[np.argmax(y2)],y2[np.argmax(y2)]+0.1), fontweight='bold', fontsize=12)
    plt.scatter(x[np.argmax(y2)], y2[np.argmax(y2)])

    plt.legend(loc='lower left',prop={'size':9, 'weight':'bold'})
    #plt.title('Acurácia do L-SVM nos conjuntos de validação',fontsize=20)
    plt.ylabel('Acurácia na Validação (%)',fontsize=18)
    plt.xlabel('Número de componentes principais (PCA)',fontsize=18)
    plt.tight_layout()
    plt.savefig(Path('DCT_LSVM_Val_Acc.png'))
    plt.show()

def plotQSVM():
    QSVM_ValPerfAugmPatLvDivImages = Path('results/Q-SVM_ValidPerformance_DCTData_AugmPatLvDivImages.csv')
    QSVM_ValPerfAugmRndDivImages = Path('results/Q-SVM_ValidPerformance_DCTData_AugmRndDivImages.csv')
    QSVM_ValPerfPatLvDivImages = Path('results/Q-SVM_ValidPerformance_DCTData_PatLvDivImages.csv')
    QSVM_ValPerfRndDivImages = Path('results/Q-SVM_ValidPerformance_DCTData_RndDivImages.csv')
    QSVM_ValPerfAugmPatLvDivImages = pd.read_csv(QSVM_ValPerfAugmPatLvDivImages, index_col=0)
    QSVM_ValPerfAugmRndDivImages = pd.read_csv(QSVM_ValPerfAugmRndDivImages, index_col=0)
    QSVM_ValPerfPatLvDivImages = pd.read_csv(QSVM_ValPerfPatLvDivImages, index_col=0)
    QSVM_ValPerfRndDivImages = pd.read_csv(QSVM_ValPerfRndDivImages, index_col=0)
    x = QSVM_ValPerfAugmPatLvDivImages['Features'].values
    y0 = QSVM_ValPerfAugmPatLvDivImages['Q-SVM_acc'].values
    y1 = QSVM_ValPerfAugmRndDivImages['Q-SVM_acc'].values
    y2 = QSVM_ValPerfPatLvDivImages['Q-SVM_acc'].values
    y3 = QSVM_ValPerfRndDivImages['Q-SVM_acc'].values
    
    plt.figure(figsize=(10, 6))
    ###############################################
    plt.plot(x, y3, label='Divisão Aleatória')
    plt.annotate(f'{y3[np.argmax(y3)]}%'.replace('.',','),
                   xy=(x[np.argmax(y3)]-35,y3[np.argmax(y3)]+0.2), fontweight='bold', fontsize=12)
    plt.scatter(x[np.argmax(y3)], y3[np.argmax(y3)])
    ###############################################
    plt.plot(x, y1, label='Divisão Aleatória com Data Augmentation')
    plt.annotate(f'{y1[np.argmax(y1)]}%'.replace('.',','),
                   xy=(x[np.argmax(y1)]-35,y1[np.argmax(y1)]+0.2), fontweight='bold', fontsize=12)
    plt.scatter(x[np.argmax(y1)], y1[np.argmax(y1)])
    ###############################################
    plt.plot(x, y0, label='Divisão por Paciente com Data Augmentation')
    plt.annotate(f'{y0[np.argmax(y0)]}%'.replace('.',','),
                   xy=(x[np.argmax(y0)]-35,y0[np.argmax(y0)]+0.2), fontweight='bold', fontsize=12)
    plt.scatter(x[np.argmax(y0)], y0[np.argmax(y0)])
    ###############################################
    plt.plot(x, y2, label='Divisão por Paciente')
    plt.annotate(f'{y2[np.argmax(y2)]}%'.replace('.',','),
                   xy=(x[np.argmax(y2)]-35,y2[np.argmax(y2)]-0.6), fontweight='bold', fontsize=12)
    plt.scatter(x[np.argmax(y2)], y2[np.argmax(y2)])

    plt.legend(loc='lower right',prop={'size': 14, 'weight':'bold'})
    #plt.xlim((0,125))
    #plt.title('Acurácia do Q-SVM nos conjuntos de validação',fontsize=20)
    plt.ylabel('Acurácia na Validação (%)',fontsize=18)
    plt.xlabel('Número de componentes principais (PCA)',fontsize=18)
    plt.tight_layout()
    plt.savefig(Path('DCT_QSVM_Val_Acc.png'))
    plt.show()

def plotPSVM():
    PSVM_ValAugmPatLvDivImages = Path('results/P-SVM_ValidPerformance_DCTData_AugmPatLvDivImages.csv')
    PSVM_ValAugmRndDivImages = Path('results/P-SVM_ValidPerformance_DCTData_AugmRndDivImages.csv')
    PSVM_ValPatLvDivImages = Path('results/P-SVM_ValidPerformance_DCTData_PatLvDivImages.csv')
    PSVM_ValRndDivImages = Path('results/P-SVM_ValidPerformance_DCTData_RndDivImages.csv')
    PSVM_ValAugmPatLvDivImages = pd.read_csv(PSVM_ValAugmPatLvDivImages, index_col=0)
    PSVM_ValAugmRndDivImages = pd.read_csv(PSVM_ValAugmRndDivImages, index_col=0)
    PSVM_ValPatLvDivImages = pd.read_csv(PSVM_ValPatLvDivImages, index_col=0)
    PSVM_ValRndDivImages = pd.read_csv(PSVM_ValRndDivImages, index_col=0)

    x = PSVM_ValAugmPatLvDivImages['Features'].values
    y0 = PSVM_ValAugmPatLvDivImages['P-SVM_acc'].values
    y1 = PSVM_ValAugmRndDivImages['P-SVM_acc'].values
    y2 = PSVM_ValPatLvDivImages['P-SVM_acc'].values
    y3 = PSVM_ValRndDivImages['P-SVM_acc'].values

    plt.figure(figsize=(10, 6))
    ###############################################
    plt.plot(x, y3, label='Divisão Aleatória')
    plt.annotate(f'{y3[np.argmax(y3)]:.2f}%'.replace('.',','),
                   xy=(x[np.argmax(y3)],y3[np.argmax(y3)]+0.1), fontweight='bold', fontsize=12)
    plt.scatter(x[np.argmax(y3)], y3[np.argmax(y3)])
    ###############################################
    plt.plot(x, y1, label='Divisão Aleatória com Data Augmentation')
    plt.annotate(f'{y1[np.argmax(y1)]:.2f}%'.replace('.',','),
                   xy=(x[np.argmax(y1)]-35,y1[np.argmax(y1)]-0.6), fontweight='bold', fontsize=12)
    plt.scatter(x[np.argmax(y1)], y1[np.argmax(y1)])
    ###############################################
    plt.plot(x, y2, label='Divisão por Paciente')
    plt.annotate(f'{y2[np.argmax(y2)]:.2f}%'.replace('.',','),
                   xy=(x[np.argmax(y2)],y2[np.argmax(y2)]+0.1), fontweight='bold', fontsize=12)
    plt.scatter(x[np.argmax(y2)], y2[np.argmax(y2)])
    ###############################################
    plt.plot(x, y0, label='Divisão por Paciente com Data Augmentation')
    plt.annotate(f'{y0[np.argmax(y0)]:.2f}%'.replace('.',','),
                   xy=(x[np.argmax(y0)],y0[np.argmax(y0)]-0.6), fontweight='bold', fontsize=12)
    plt.scatter(x[np.argmax(y0)], y0[np.argmax(y0)])
    
    
    plt.legend(loc='lower center',prop={'size': 14, 'weight':'bold'})
    #plt.title('Acurácia do P-SVM nos conjuntos de validação',fontsize=20)
    plt.ylabel('Acurácia na Validação (%)',fontsize=18)
    plt.xlabel('Número de componentes principais (PCA)',fontsize=18)
    plt.tight_layout()
    plt.savefig(Path('DCT_PSVM_Val_Acc.png'))
    plt.show()

def plotRSVM():
    RSVM_ValAugmPatLvDivImages = Path('results/R-SVM_ValidPerformance_DCTData_AugmPatLvDivImages.csv')
    RSVM_ValAugmRndDivImages = Path('results/R-SVM_ValidPerformance_DCTData_AugmRndDivImages.csv')
    RSVM_ValPatLvDivImages = Path('results/R-SVM_ValidPerformance_DCTData_PatLvDivImages.csv')
    RSVM_ValRndDivImages = Path('results/R-SVM_ValidPerformance_DCTData_RndDivImages.csv')
    RSVM_ValAugmPatLvDivImages = pd.read_csv(RSVM_ValAugmPatLvDivImages, index_col=0)
    RSVM_ValAugmRndDivImages = pd.read_csv(RSVM_ValAugmRndDivImages, index_col=0)
    RSVM_ValPatLvDivImages = pd.read_csv(RSVM_ValPatLvDivImages, index_col=0)
    RSVM_ValRndDivImages = pd.read_csv(RSVM_ValRndDivImages, index_col=0)

    x = RSVM_ValAugmPatLvDivImages['Features'].values
    y0 = RSVM_ValAugmPatLvDivImages['R-SVM_acc'].values
    y1 = RSVM_ValAugmRndDivImages['R-SVM_acc'].values
    y2 = RSVM_ValPatLvDivImages['R-SVM_acc'].values
    y3 = RSVM_ValRndDivImages['R-SVM_acc'].values

    plt.figure(figsize=(10, 6))
    ###############################################
    plt.plot(x, y3, label='Divisão Aleatória')
    plt.annotate(f'{y3[np.argmax(y3)]}%'.replace('.',','),
                   xy=(x[np.argmax(y3)],y3[np.argmax(y3)]+0.1), fontweight='bold', fontsize=12)
    plt.scatter(x[np.argmax(y3)], y3[np.argmax(y3)])
    ###############################################
    plt.plot(x, y1, label='Divisão Aleatória com Data Augmentation')
    plt.annotate(f'{y1[np.argmax(y1)]}%'.replace('.',','),
                   xy=(x[np.argmax(y1)],y1[np.argmax(y1)]+0.1), fontweight='bold', fontsize=12)
    plt.scatter(x[np.argmax(y1)], y1[np.argmax(y1)])
    ###############################################
    plt.plot(x, y2, label='Divisão por Paciente')
    plt.annotate(f'{y2[np.argmax(y2)]}%'.replace('.',','),
                   xy=(x[np.argmax(y2)],y2[np.argmax(y2)]+0.1), fontweight='bold', fontsize=12)
    plt.scatter(x[np.argmax(y2)], y2[np.argmax(y2)])
    ###############################################
    plt.plot(x, y0, label='Divisão por Paciente com Data Augmentation')
    plt.annotate(f'{y0[np.argmax(y0)]}%'.replace('.',','),
                   xy=(x[np.argmax(y0)],y0[np.argmax(y0)]-0.3), fontweight='bold', fontsize=12)
    plt.scatter(x[np.argmax(y0)], y0[np.argmax(y0)])

    plt.legend(loc='lower center',prop={'size': 10, 'weight':'bold'})
    #plt.title('Acurácia do R-SVM nos conjuntos de validação',fontsize=20)
    plt.ylabel('Acurácia na Validação (%)',fontsize=18)
    plt.xlabel('Número de componentes principais (PCA)',fontsize=18)
    plt.tight_layout()
    plt.savefig(Path('DCT_RSVM_Val_Acc.png'))
    plt.show()

def plotSigSVM():
    SigSVM_ValAugmPatLvDivImages = Path('results/Sig-SVM_ValidPerformance_DCTData_AugmPatLvDivImages.csv')
    SigSVM_ValAugmRndDivImages = Path('results/Sig-SVM_ValidPerformance_DCTData_AugmRndDivImages.csv')
    SigSVM_ValPatLvDivImages = Path('results/Sig-SVM_ValidPerformance_DCTData_PatLvDivImages.csv')
    SigSVM_ValRndDivImages = Path('results/Sig-SVM_ValidPerformance_DCTData_RndDivImages.csv')
    SigSVM_ValAugmPatLvDivImages = pd.read_csv(SigSVM_ValAugmPatLvDivImages, index_col=0)
    SigSVM_ValAugmRndDivImages = pd.read_csv(SigSVM_ValAugmRndDivImages, index_col=0)
    SigSVM_ValPatLvDivImages = pd.read_csv(SigSVM_ValPatLvDivImages, index_col=0)
    SigSVM_ValRndDivImages = pd.read_csv(SigSVM_ValRndDivImages, index_col=0)

    x = SigSVM_ValAugmPatLvDivImages['Features'].values
    y0 = SigSVM_ValAugmPatLvDivImages['S-SVM_acc'].values
    y1 = SigSVM_ValAugmRndDivImages['S-SVM_acc'].values
    y2 = SigSVM_ValPatLvDivImages['S-SVM_acc'].values
    y3 = SigSVM_ValRndDivImages['S-SVM_acc'].values

    plt.figure(figsize=(15, 6))
    ###############################################
    plt.plot(x, y3, label='Divisão Aleatória')
    plt.annotate(f'{y3[np.argmax(y3)]:.2f}%'.replace('.',','),
                   xy=(x[np.argmax(y3)],y3[np.argmax(y3)]+0.2), fontweight='bold', fontsize=12)
    plt.scatter(x[np.argmax(y3)], y3[np.argmax(y3)])
    ###############################################
    plt.plot(x, y1, label='Divisão Aleatória com Data Augmentation') 
    plt.annotate(f'{y1[np.argmax(y1)]:.2f}%'.replace('.',','),
                   xy=(x[np.argmax(y1)],y1[np.argmax(y1)]+0.2), fontweight='bold', fontsize=12)
    plt.scatter(x[np.argmax(y1)], y1[np.argmax(y1)])
    ###############################################
    plt.plot(x, y2, label='Divisão por Paciente')
    plt.annotate(f'{y2[np.argmax(y2)]:.2f}%'.replace('.',','),
                   xy=(x[np.argmax(y2)],y2[np.argmax(y2)]-0.8), fontweight='bold', fontsize=12)
    plt.scatter(x[np.argmax(y2)], y2[np.argmax(y2)])
    ###############################################
    plt.plot(x, y0, label='Divisão por Paciente com Data Augmentation')
    plt.annotate(f'{y0[np.argmax(y0)]:.2f}%'.replace('.',','),
                   xy=(x[np.argmax(y0)],y0[np.argmax(y0)]+0.2), fontweight='bold', fontsize=12)
    plt.scatter(x[np.argmax(y0)], y0[np.argmax(y0)])

    plt.legend(loc='lower right',prop={'size': 14, 'weight':'bold'})
    #plt.title('Acurácia do Sig-SVM nos conjuntos de validação',fontsize=20)
    plt.ylabel('Acurácia na Validação (%)',fontsize=18)
    plt.xlabel('Número de componentes principais (PCA)',fontsize=18)
    plt.tight_layout()
    plt.savefig(Path('DCT_SigSVM_Val_Acc.png'))
    plt.show()

def plotKNN():
    KNN_ValAugmPatLvDivImages = Path('results/KNN_ValidPerformance_DCTData_AugmPatLvDivImages.csv')
    KNN_ValAugmRndDivImages = Path('results/KNN_ValidPerformance_DCTData_AugmRndDivImages.csv')
    KNN_ValPatLvDivImages = Path('results/KNN_ValidPerformance_DCTData_PatLvDivImages.csv')
    KNN_ValrndDivImages = Path('results/KNN_ValidPerformance_DCTData_rndDivImages.csv')
    KNN_ValAugmPatLvDivImages = pd.read_csv(KNN_ValAugmPatLvDivImages, index_col=0)
    KNN_ValAugmRndDivImages = pd.read_csv(KNN_ValAugmRndDivImages, index_col=0)
    KNN_ValPatLvDivImages = pd.read_csv(KNN_ValPatLvDivImages, index_col=0)
    KNN_ValrndDivImages = pd.read_csv(KNN_ValrndDivImages, index_col=0)

    cols = ['Features']+[f'{i}-KNN_acc' for i in range(5,21)]
    KNN_ValAugmPatLvDivImages = KNN_ValAugmPatLvDivImages[cols]
    KNN_ValAugmRndDivImages = KNN_ValAugmRndDivImages[cols]
    KNN_ValPatLvDivImages = KNN_ValPatLvDivImages[cols]
    KNN_ValrndDivImages = KNN_ValrndDivImages[cols]

    def printKNNTable(df):
        lines = []
        pcaRange = [50,100,150,200,300,400,600,800,1024]
        vals = df.values[:,1:]
        boldPoints = []
        for col in range(vals.shape[1]):
            boldPoints.append((col, pcaRange[np.argmax(vals[:,col])]))
        lines.append('\\begin{tabular}{c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c}')
        lines.append('PCA & k=5 & k=6 & k=7 & k=8 & k=9 & k=10 & k=11 & k=12 & k=13 & k=14 & k=15 & k=16 & k=17 & k=18 & k=19 & k=20 \\\\')
        lines.append('\\hline')
        for n, pcaComps in enumerate(pcaRange):
            v = df.iloc[n].values[1:].tolist()
            v = list(map(lambda n:f'{n:.2f}'.replace('.',','), v))
            for bp in boldPoints:
                if bp[1] == pcaComps:
                    v[bp[0]] = f'\\textbf{{{v[bp[0]]}}}'
            line = f'{pcaComps} & {v[0]} & {v[1]} & {v[2]} & {v[3]} & {v[4]} & {v[5]} & {v[6]} & {v[7]} & {v[8]} & {v[9]} & {v[10]} & {v[11]} & {v[12]} & {v[13]} & {v[14]} & {v[15]} \\\\'
            lines.append(line)
        lines.append('\\hline')
        lines.append('\\end{tabular}')
        print('\n'.join(lines))
    
    #printKNNTable(KNN_ValAugmPatLvDivImages)
    #printKNNTable(KNN_ValAugmRndDivImages)
    #printKNNTable(KNN_ValPatLvDivImages)
    #printKNNTable(KNN_ValrndDivImages)
    return None

def plotRF():
    RF_ValAugmPatLvDivImages = Path('results/RF_ValidPerformance_DCTData_AugmPatLvDivImages.csv')
    RF_ValAugmRndDivImages = Path('results/RF_ValidPerformance_DCTData_AugmRndDivImages.csv')
    RF_ValPatLvDivImages = Path('results/RF_ValidPerformance_DCTData_PatLvDivImages.csv')
    RF_ValRndDivImages = Path('results/RF_ValidPerformance_DCTData_RndDivImages.csv')
    RF_ValAugmPatLvDivImages = pd.read_csv(RF_ValAugmPatLvDivImages, index_col=0)
    RF_ValAugmRndDivImages = pd.read_csv(RF_ValAugmRndDivImages, index_col=0)
    RF_ValPatLvDivImages = pd.read_csv(RF_ValPatLvDivImages, index_col=0)
    RF_ValRndDivImages = pd.read_csv(RF_ValRndDivImages, index_col=0)
    
    cols = ['Features']+[f'{i}-RF_acc' for i in range(50,160,20)]
    RF_ValAugmPatLvDivImages = RF_ValAugmPatLvDivImages[cols]
    RF_ValAugmRndDivImages = RF_ValAugmRndDivImages[cols]
    RF_ValPatLvDivImages = RF_ValPatLvDivImages[cols]
    RF_ValRndDivImages = RF_ValRndDivImages[cols]

    def printRFTable(df):
        lines = []
        pcaRange = [50,100,150,200,300,400,600,800,1024]
        vals = df.values[:,1:]
        boldPoints = []
        for col in range(vals.shape[1]):
            boldPoints.append((col, pcaRange[np.argmax(vals[:,col])]))
        lines.append('\\begin{tabular}{c|cccccc}')
        lines.append('PCA & 50 árvores & 70 árvores & 90 árvores & 110 árvores & 130 árvores & 150 árvores \\\\')
        lines.append('\\hline')
        for n, pcaComps in enumerate(pcaRange):
            v = df.iloc[n].values[1:].tolist()
            v = list(map(lambda n:f'{n:.2f}'.replace('.',','), v))
            for bp in boldPoints:
                if bp[1] == pcaComps:
                    v[bp[0]] = f'\\textbf{{{v[bp[0]]}}}'
            line = f'{pcaComps} & {v[0]}\\% & {v[1]}\\% & {v[2]}\\% & {v[3]}\\% & {v[4]}\\% & {v[5]}\\% \\\\'
            lines.append(line)
        lines.append('\\hline')
        lines.append('\\end{tabular}')
        print('\n'.join(lines))

    #printRFTable(RF_ValAugmPatLvDivImages)
    #printRFTable(RF_ValAugmRndDivImages)
    #printRFTable(RF_ValPatLvDivImages)
    #printRFTable(RF_ValRndDivImages)
    return None

def plotNN():
    NN_augmPatLvDiv = Path('results/run_augmPatLvDiv_HL-4_NEU-512-Acti-tanh-tag-val_acc.csv')
    NN_augmRndDiv = Path('results/run_augmRndDiv_HL-4_NEU-1024-Acti-sigmoid-tag-val_acc.csv')
    NN_patLvDiv = Path('results/run_patLvDiv_HL-3_NEU-32-Acti-relu-tag-val_acc.csv')
    NN_rndDiv = Path('results/run_rndDiv_HL-2_NEU-512-Acti-relu-tag-val_acc.csv')

    NN_augmPatLvDiv = pd.read_csv(NN_augmPatLvDiv,index_col=0)
    NN_augmRndDiv = pd.read_csv(NN_augmRndDiv,index_col=0)
    NN_patLvDiv = pd.read_csv(NN_patLvDiv,index_col=0)
    NN_rndDiv = pd.read_csv(NN_rndDiv,index_col=0)

    x = NN_augmPatLvDiv['Step'].values
    y0 = NN_augmPatLvDiv['Value'].values * 100.0
    y1 = NN_augmRndDiv['Value'].values * 100.0
    y2 = NN_patLvDiv['Value'].values * 100.0
    y3 = NN_rndDiv['Value'].values * 100.0

    plt.figure(figsize=(10, 6))
    ###############################################
    plt.plot(x, y1, label='Divisão Aleatória com Data Augmentation')
    plt.annotate(f'{np.round(y1[np.argmax(y1)], decimals=2):.2f}%'.replace('.',','),
                   xy=(x[np.argmax(y1)],y1[np.argmax(y1)]+0.1), fontweight='bold', fontsize=12)
    plt.scatter(x[np.argmax(y1)], y1[np.argmax(y1)])
    ###############################################
    plt.plot(x, y3, label='Divisão Aleatória')
    plt.annotate(f'{np.round(y3[np.argmax(y3)], decimals=2):.2f}%'.replace('.',','),
                   xy=(x[np.argmax(y3)]-5,y3[np.argmax(y3)]+0.1), fontweight='bold', fontsize=12)
    plt.scatter(x[np.argmax(y3)], y3[np.argmax(y3)])
    ###############################################
    plt.plot(x, y0, label='Divisão por Paciente com Data Augmentation')
    plt.annotate(f'{np.round(y0[np.argmax(y0)], decimals=2):.2f}%'.replace('.',','),
                   xy=(x[np.argmax(y0)],y0[np.argmax(y0)]+0.1), fontweight='bold', fontsize=12)
    plt.scatter(x[np.argmax(y0)], y0[np.argmax(y0)])
    ###############################################
    plt.plot(x, y2, label='Divisão por Paciente')
    plt.annotate(f'{np.round(y2[np.argmax(y2)], decimals=2):.2f}%'.replace('.',','),
                   xy=(x[np.argmax(y2)],y2[np.argmax(y2)]+0.1), fontweight='bold', fontsize=12)
    plt.scatter(x[np.argmax(y2)], y2[np.argmax(y2)])

    plt.legend(loc='lower center',prop={'size': 14, 'weight':'bold'})
    #plt.title('Acurácia das Redes Neurais nos conjuntos de validação',fontsize=20)
    plt.ylabel('Acurácia na Validação (%)',fontsize=18)
    plt.xlabel('Épocas de treinamento',fontsize=18)
    #plt.xlim((-5,180))
    plt.tight_layout()
    plt.savefig(Path('DCT_NN_Val_Acc.png'))
    plt.show()


if __name__ == '__main__':    
    #plotLSVM()
    #plotQSVM()
    #plotPSVM()
    #plotRSVM()
    #plotSigSVM()
    #plotKNN()
    #plotRF()
    #plotNN()
    
    print(f"\nEnd Script!\n{'#'*50}")
