from pathlib import Path
import multiprocessing
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import svm, neighbors, ensemble
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

augmPatLvDiv_TRAIN = Path('data/AugmPatLvDiv_TRAIN-ContourData_160-Features_20000-images.csv')
augmPatLvDiv_VALIDATION = Path('data/AugmPatLvDiv_VALIDATION-ContourData_160-Features_5000-images.csv')
augmRndDiv_TRAIN = Path('data/AugmRndDiv_TRAIN-ContourData_160-Features_20000-images.csv')
augmRndDiv_VALIDATION = Path('data/AugmRndDiv_VALIDATION-ContourData_160-Features_5000-images.csv')
patLvDiv_TEST = Path('data/PatLvDiv_TEST-ContourData_160-Features_550-images.csv')
patLvDiv_TRAIN = Path('data/PatLvDiv_TRAIN-ContourData_160-Features_7224-images.csv')
patLvDiv_VALIDATION = Path('data/PatLvDiv_VALIDATION-ContourData_160-Features_2887-images.csv')
rndDiv_TEST = Path('data/rndDiv_TEST-ContourData_160-Features_1066-images.csv')
rndDiv_TRAIN = Path('data/rndDiv_TRAIN-ContourData_160-Features_6398-images.csv')
rndDiv_VALIDATION = Path('data/rndDiv_VALIDATION-ContourData_160-Features_3197-images.csv')

from B_SKLearnClassifier_ContourFeats import balanceDataset, computeStats, \
                                             Xy_Split, validateCLF

from B_SKLearnClassifier_ContourFeats import L_SVM_CLF, Q_SVM_CLF, Poly_SVM_CLF, RBF_SVM_CLF, Sigm_SVM_CLF,\
                                             KNN_CLF, RF_CLF


def printTable(dfPath):
    df = pd.read_csv(dfPath)
    l00 = '\\begin{tabular}{l|c|ccccc}'
    l01 = '\\multicolumn{1}{c|}{Conj.Treino} & Conj.Teste & Acurácia & Precisão & Sensibilidade & Especificidade & \\textit{F1-score} \\\\'
    l02 = '\\cline{1-7}'
    
    vals = df.iloc[0].values[2:8].tolist()
    vals.pop(2) #Remove Balanced Accuracy
    vals = list(map(lambda n:f'{n:.2f}'.replace('.',','), vals))
    l03 = f'Divisão por paciente (com augm.) & \\multirow{{2}}{{*}}{{\\textit{{patLvDiv\\_test}}}} & {vals[1]}\\% & {vals[2]}\\% & {vals[3]}\\% & {vals[4]}\\% & {vals[0]}\\% \\\\'

    vals = df.iloc[2].values[2:8].tolist()
    vals.pop(2) #Remove Balanced Accuracy
    vals = list(map(lambda n:f'{n:.2f}'.replace('.',','), vals))
    l04 = f'Divisão por paciente (sem augm.) &                                          & {vals[1]}\\% & {vals[2]}\\% & {vals[3]}\\% & {vals[4]}\\% & {vals[0]}\\% \\\\'

    l05 = '\\cline{1-7}'

    vals = df.iloc[1].values[8:].tolist()
    vals.pop(2) #Remove Balanced Accuracy
    vals = list(map(lambda n:f'{n:.2f}'.replace('.',','), vals))
    l06 = f'Divisão aleatória (com augm.)    & \\multirow{{2}}{{*}}{{\\textit{{rndDiv\\_test}}}}   & {vals[1]}\\% & {vals[2]}\\% & {vals[3]}\\% & {vals[4]}\\% & {vals[0]}\\% \\\\'

    vals = df.iloc[3].values[8:].tolist()
    vals.pop(2) #Remove Balanced Accuracy
    vals = list(map(lambda n:f'{n:.2f}'.replace('.',','), vals))
    l07 = f'Divisão aleatória (sem augm.)    &                                          & {vals[1]}\\% & {vals[2]}\\% & {vals[3]}\\% & {vals[4]}\\% & {vals[0]}\\% \\\\'

    l08 = '\\cline{1-7}'
    l09 = '\\end{tabular}'

    print('\n'.join([l00,l01,l02,l03,l04,l05,l06,l07,l08,l09]))

##################################################################################################################
def testSVMClassifier(testID, train_df, funcPointer, funcParams, results_df):
    clfName = testID.split('-')[0]
    collectedData = {}
    collectedData['testID'] = testID.split('-')[1]
    y_true, y_pred = funcPointer(train_df,
                                 balanceDataset(patLvDiv_TEST),
                                 params=funcParams['params'],
                                 PCA_Feats=funcParams['PCA_Feats'])
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=['HEM', 'ALL']).ravel()
    sens, spec, acc, prec = computeStats(tn, fp, fn, tp)
    collectedData[f'test-patLvDiv_{clfName}_sens'] = sens
    collectedData[f'test-patLvDiv_{clfName}_spec'] = spec
    collectedData[f'test-patLvDiv_{clfName}_acc'] = acc
    collectedData[f'test-patLvDiv_{clfName}_prec'] = prec
    collectedData[f'test-patLvDiv_{clfName}_F1sco'] = np.round(2*((prec*sens)/(prec+sens)), decimals=2)
    collectedData[f'test-patLvDiv_{clfName}_balAcc'] = np.round((sens+spec)/2, decimals=2)
    y_true, y_pred = funcPointer(train_df,
                                 balanceDataset(rndDiv_TEST),
                                 params=funcParams['params'],
                                 PCA_Feats=funcParams['PCA_Feats'])
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=['HEM', 'ALL']).ravel()
    sens, spec, acc, prec = computeStats(tn, fp, fn, tp)
    collectedData[f'test-rndDiv_{clfName}_sens'] = sens
    collectedData[f'test-rndDiv_{clfName}_spec'] = spec
    collectedData[f'test-rndDiv_{clfName}_acc'] = acc
    collectedData[f'test-rndDiv_{clfName}_prec'] = prec
    collectedData[f'test-rndDiv_{clfName}_F1sco'] = np.round(2*((prec*sens)/(prec+sens)), decimals=2)
    collectedData[f'test-rndDiv_{clfName}_balAcc'] = np.round((sens+spec)/2, decimals=2)
    return results_df.append(collectedData, ignore_index=True)
##################################################################################################################

def testLSVM():
    lsvm_df = pd.DataFrame()
    lsvm_df = testSVMClassifier(testID='LSVM-augmPatLvDiv',
                                train_df=augmPatLvDiv_TRAIN,
                                funcPointer=L_SVM_CLF,
                                funcParams={'params':{'C':1.0, 'tol':0.001, 'max_iter':-1}, 'PCA_Feats':None},
                                results_df=lsvm_df)
    lsvm_df = testSVMClassifier(testID='LSVM-augmRndDiv',
                                train_df=augmRndDiv_TRAIN,
                                funcPointer=L_SVM_CLF,
                                funcParams={'params':{'C':1.0, 'tol':0.001, 'max_iter':-1}, 'PCA_Feats':None},
                                results_df=lsvm_df)
    lsvm_df = testSVMClassifier(testID='LSVM-patLvDiv',
                                train_df=patLvDiv_TRAIN,
                                funcPointer=L_SVM_CLF,
                                funcParams={'params':{'C':1.0, 'tol':0.001, 'max_iter':-1}, 'PCA_Feats':140},
                                results_df=lsvm_df)
    lsvm_df = testSVMClassifier(testID='LSVM-rndDiv',
                                train_df=rndDiv_TRAIN,
                                funcPointer=L_SVM_CLF,
                                funcParams={'params':{'C':1.0, 'tol':0.001, 'max_iter':-1}, 'PCA_Feats':None},
                                results_df=lsvm_df)
    cols = lsvm_df.columns.tolist()
    cols.remove('testID')
    lsvm_df = lsvm_df[['testID'] + cols]
    lsvm_df.to_csv(Path('results/L-SVM_TestPerformance_ContourData.csv'))

def testQSVM():
    qsvm_df = pd.DataFrame()
    qsvm_df = testSVMClassifier(testID='QSVM-augmPatLvDiv',
                                train_df=augmPatLvDiv_TRAIN,
                                funcPointer=Q_SVM_CLF,
                                funcParams={'params':{'C':1.0, 'coef0':0.0,'tol':0.001, 'max_iter':-1}, 'PCA_Feats':110},
                                results_df=qsvm_df)
    qsvm_df = testSVMClassifier(testID='QSVM-augmRndDiv',
                                train_df=augmRndDiv_TRAIN,
                                funcPointer=Q_SVM_CLF,
                                funcParams={'params':{'C':1.0, 'coef0':0.0,'tol':0.001, 'max_iter':-1}, 'PCA_Feats':160},
                                results_df=qsvm_df)
    qsvm_df = testSVMClassifier(testID='QSVM-patLvDiv',
                                train_df=patLvDiv_TRAIN,
                                funcPointer=Q_SVM_CLF,
                                funcParams={'params':{'C':1.0, 'coef0':0.0,'tol':0.001, 'max_iter':-1}, 'PCA_Feats':120},
                                results_df=qsvm_df)
    qsvm_df = testSVMClassifier(testID='QSVM-rndDiv',
                                train_df=rndDiv_TRAIN,
                                funcPointer=Q_SVM_CLF,
                                funcParams={'params':{'C':1.0, 'coef0':0.0,'tol':0.001, 'max_iter':-1}, 'PCA_Feats':20},
                                results_df=qsvm_df)
    cols = qsvm_df.columns.tolist()
    cols.remove('testID')
    qsvm_df = qsvm_df[['testID'] + cols]
    qsvm_df.to_csv(Path('results/Q-SVM_TestPerformance_ContourData.csv'))

def testPSVM():
    psvm_df = pd.DataFrame()
    psvm_df = testSVMClassifier(testID='PSVM-augmPatLvDiv',
                                train_df=augmPatLvDiv_TRAIN,
                                funcPointer=Poly_SVM_CLF,
                                funcParams={'params':{'C':1.0, 'coef0':0.0,'tol':0.001, 'max_iter':1e6}, 'PCA_Feats':130},
                                results_df=psvm_df)
    psvm_df = testSVMClassifier(testID='PSVM-augmRndDiv',
                                train_df=augmRndDiv_TRAIN,
                                funcPointer=Poly_SVM_CLF,
                                funcParams={'params':{'C':1.0, 'coef0':0.0,'tol':0.001, 'max_iter':1e6}, 'PCA_Feats':140},
                                results_df=psvm_df)
    psvm_df = testSVMClassifier(testID='PSVM-patLvDiv',
                                train_df=patLvDiv_TRAIN,
                                funcPointer=Poly_SVM_CLF,
                                funcParams={'params':{'C':1.0, 'coef0':0.0,'tol':0.001, 'max_iter':1e6}, 'PCA_Feats':150},
                                results_df=psvm_df)
    psvm_df = testSVMClassifier(testID='PSVM-rndDiv',
                                train_df=rndDiv_TRAIN,
                                funcPointer=Poly_SVM_CLF,
                                funcParams={'params':{'C':1.0, 'coef0':0.0,'tol':0.001, 'max_iter':1e6}, 'PCA_Feats':None},
                                results_df=psvm_df)
    cols = psvm_df.columns.tolist()
    cols.remove('testID')
    psvm_df = psvm_df[['testID'] + cols]
    psvm_df.to_csv(Path('results/P-SVM_TestPerformance_ContourData.csv'))

def testRSVM():
    rsvm_df = pd.DataFrame()
    rsvm_df = testSVMClassifier(testID='RSVM-augmPatLvDiv',
                                train_df=augmPatLvDiv_TRAIN,
                                funcPointer=RBF_SVM_CLF,
                                funcParams={'params':{'C':1.0,'tol':0.001, 'max_iter':-1}, 'PCA_Feats':160},
                                results_df=rsvm_df)
    rsvm_df = testSVMClassifier(testID='RSVM-augmRndDiv',
                                train_df=augmRndDiv_TRAIN,
                                funcPointer=RBF_SVM_CLF,
                                funcParams={'params':{'C':1.0,'tol':0.001, 'max_iter':-1}, 'PCA_Feats':160},
                                results_df=rsvm_df)
    rsvm_df = testSVMClassifier(testID='RSVM-patLvDiv',
                                train_df=patLvDiv_TRAIN,
                                funcPointer=RBF_SVM_CLF,
                                funcParams={'params':{'C':1.0,'tol':0.001, 'max_iter':-1}, 'PCA_Feats':160},
                                results_df=rsvm_df)
    rsvm_df = testSVMClassifier(testID='RSVM-rndDiv',
                                train_df=rndDiv_TRAIN,
                                funcPointer=RBF_SVM_CLF,
                                funcParams={'params':{'C':1.0,'tol':0.001, 'max_iter':-1}, 'PCA_Feats':140},
                                results_df=rsvm_df)
    cols = rsvm_df.columns.tolist()
    cols.remove('testID')
    rsvm_df = rsvm_df[['testID'] + cols]
    rsvm_df.to_csv(Path('results/R-SVM_TestPerformance_ContourData.csv'))

def testSigSVM():
    ssvm_df = pd.DataFrame()
    ssvm_df = testSVMClassifier(testID='SigSVM-augmPatLvDiv',
                                train_df=augmPatLvDiv_TRAIN,
                                funcPointer=Sigm_SVM_CLF,
                                funcParams={'params':{'C':1.0, 'coef0':0.0,'tol':0.001, 'max_iter':-1}, 'PCA_Feats':160},
                                results_df=ssvm_df)
    ssvm_df = testSVMClassifier(testID='SigSVM-augmRndDiv',
                                train_df=augmRndDiv_TRAIN,
                                funcPointer=Sigm_SVM_CLF,
                                funcParams={'params':{'C':1.0, 'coef0':0.0,'tol':0.001, 'max_iter':-1}, 'PCA_Feats':130},
                                results_df=ssvm_df)
    ssvm_df = testSVMClassifier(testID='SigSVM-patLvDiv',
                                train_df=patLvDiv_TRAIN,
                                funcPointer=Sigm_SVM_CLF,
                                funcParams={'params':{'C':1.0, 'coef0':0.0,'tol':0.001, 'max_iter':-1}, 'PCA_Feats':150},
                                results_df=ssvm_df)
    ssvm_df = testSVMClassifier(testID='SigSVM-rndDiv',
                                train_df=rndDiv_TRAIN,
                                funcPointer=Sigm_SVM_CLF,
                                funcParams={'params':{'C':1.0, 'coef0':0.0,'tol':0.001, 'max_iter':-1}, 'PCA_Feats':160},
                                results_df=ssvm_df)
    cols = ssvm_df.columns.tolist()
    cols.remove('testID')
    ssvm_df = ssvm_df[['testID'] + cols]
    ssvm_df.to_csv(Path('results/Sig-SVM_TestPerformance_ContourData.csv'))

###############################################################################
###############################################################################
###############################################################################
###############################################################################

def testKNNClassifier(testID, train_df, funcPointer, funcParams, results_df):
    collectedData = {}
    clfName = testID.split('-')[0]
    collectedData['testID'] = testID.split('-')[1]
    y_true, y_pred = funcPointer(train_df,
                                 balanceDataset(patLvDiv_TEST),
                                 n_neighbors=funcParams['n_neighbors'],
                                 PCA_Feats=funcParams['PCA_Feats'])
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=['HEM', 'ALL']).ravel()
    sens, spec, acc, prec = computeStats(tn, fp, fn, tp)
    collectedData[f'test-patLvDiv_{clfName}_sens'] = sens
    collectedData[f'test-patLvDiv_{clfName}_spec'] = spec
    collectedData[f'test-patLvDiv_{clfName}_acc'] = acc
    collectedData[f'test-patLvDiv_{clfName}_prec'] = prec
    collectedData[f'test-patLvDiv_{clfName}_F1sco'] = np.round(2*((prec*sens)/(prec+sens)), decimals=2)
    collectedData[f'test-patLvDiv_{clfName}_balAcc'] = np.round((sens+spec)/2, decimals=2)
    y_true, y_pred = funcPointer(train_df,
                                 balanceDataset(rndDiv_TEST),
                                 n_neighbors=funcParams['n_neighbors'],
                                 PCA_Feats=funcParams['PCA_Feats'])
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=['HEM', 'ALL']).ravel()
    sens, spec, acc, prec = computeStats(tn, fp, fn, tp)
    collectedData[f'test-rndDiv_{clfName}_sens'] = sens
    collectedData[f'test-rndDiv_{clfName}_spec'] = spec
    collectedData[f'test-rndDiv_{clfName}_acc'] = acc
    collectedData[f'test-rndDiv_{clfName}_prec'] = prec
    collectedData[f'test-rndDiv_{clfName}_F1sco'] = np.round(2*((prec*sens)/(prec+sens)), decimals=2)
    collectedData[f'test-rndDiv_{clfName}_balAcc'] = np.round((sens+spec)/2, decimals=2)
    return results_df.append(collectedData, ignore_index=True)

def testKNN():
    knn_df = pd.DataFrame()
    knn_df = testKNNClassifier(testID='KNN-augmPatLvDiv',
                               train_df=augmPatLvDiv_TRAIN,
                               funcPointer=KNN_CLF,
                               funcParams={'n_neighbors':19, 'PCA_Feats':160},
                               results_df=knn_df)
    knn_df = testKNNClassifier(testID='KNN-augmRndDiv',
                               train_df=augmRndDiv_TRAIN,
                               funcPointer=KNN_CLF,
                               funcParams={'n_neighbors':19, 'PCA_Feats':150},
                               results_df=knn_df)
    knn_df = testKNNClassifier(testID='KNN-patLvDiv',
                               train_df=patLvDiv_TRAIN,
                               funcPointer=KNN_CLF,
                               funcParams={'n_neighbors':19, 'PCA_Feats':10},
                               results_df=knn_df)
    knn_df = testKNNClassifier(testID='KNN-rndDiv',
                               train_df=rndDiv_TRAIN,
                               funcPointer=KNN_CLF,
                               funcParams={'n_neighbors':19, 'PCA_Feats':160},
                               results_df=knn_df)
    cols = knn_df.columns.tolist()
    cols.remove('testID')
    knn_df = knn_df[['testID'] + cols]
    knn_df.to_csv(Path('results/KNN_TestPerformance_ContourData.csv'))


###############################################################################
###############################################################################
###############################################################################
###############################################################################


def testRFClassifier(testID, train_df, funcPointer, funcParams, results_df):
        collectedData = {}
        clfName = testID.split('-')[0]
        collectedData['testID'] = testID.split('-')[1]
        y_true, y_pred = funcPointer(train_df,
                                     balanceDataset(patLvDiv_TEST),
                                     n_estimators=funcParams['n_estimators'],
                                     PCA_Feats=funcParams['PCA_Feats'])
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=['HEM', 'ALL']).ravel()
        sens, spec, acc, prec = computeStats(tn, fp, fn, tp)
        collectedData[f'test-patLvDiv_{clfName}_sens'] = sens
        collectedData[f'test-patLvDiv_{clfName}_spec'] = spec
        collectedData[f'test-patLvDiv_{clfName}_acc'] = acc
        collectedData[f'test-patLvDiv_{clfName}_prec'] = prec
        collectedData[f'test-patLvDiv_{clfName}_F1sco'] = np.round(2*((prec*sens)/(prec+sens)), decimals=2)
        collectedData[f'test-patLvDiv_{clfName}_balAcc'] = np.round((sens+spec)/2, decimals=2)
        y_true, y_pred = funcPointer(train_df,
                                     balanceDataset(rndDiv_TEST),
                                     n_estimators=funcParams['n_estimators'],
                                     PCA_Feats=funcParams['PCA_Feats'])
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=['HEM', 'ALL']).ravel()
        sens, spec, acc, prec = computeStats(tn, fp, fn, tp)
        collectedData[f'test-rndDiv_{clfName}_sens'] = sens
        collectedData[f'test-rndDiv_{clfName}_spec'] = spec
        collectedData[f'test-rndDiv_{clfName}_acc'] = acc
        collectedData[f'test-rndDiv_{clfName}_prec'] = prec
        collectedData[f'test-rndDiv_{clfName}_F1sco'] = np.round(2*((prec*sens)/(prec+sens)), decimals=2)
        collectedData[f'test-rndDiv_{clfName}_balAcc'] = np.round((sens+spec)/2, decimals=2)
        return results_df.append(collectedData, ignore_index=True)

def testRF():
    rf_df = pd.DataFrame()
    rf_df = testRFClassifier(testID='RF-augmPatLvDiv',
                              train_df=augmPatLvDiv_TRAIN,
                              funcPointer=RF_CLF,
                              funcParams={'n_estimators':150, 'PCA_Feats':150},
                              results_df=rf_df)
    rf_df = testRFClassifier(testID='RF-augmRndDiv',
                              train_df=augmRndDiv_TRAIN,
                              funcPointer=RF_CLF,
                              funcParams={'n_estimators':150, 'PCA_Feats':120},
                              results_df=rf_df)
    rf_df = testRFClassifier(testID='RF-patLvDiv',
                              train_df=patLvDiv_TRAIN,
                              funcPointer=RF_CLF,
                              funcParams={'n_estimators':70, 'PCA_Feats':160},
                              results_df=rf_df)
    rf_df = testRFClassifier(testID='RF-rndDiv',
                              train_df=rndDiv_TRAIN,
                              funcPointer=RF_CLF,
                              funcParams={'n_estimators':150, 'PCA_Feats':130},
                              results_df=rf_df)
    cols = rf_df.columns.tolist()
    cols.remove('testID')
    rf_df = rf_df[['testID'] + cols]
    rf_df.to_csv(Path('results/RF_TestPerformance_ContourData.csv'))


###############################################################################
###############################################################################
###############################################################################
###############################################################################

def testNN():
    printTable(Path('results/NN_TestPerformance_ContourData.csv'))


if __name__ == '__main__':
    #Read dataframes
    augmPatLvDiv_TRAIN = pd.read_csv(augmPatLvDiv_TRAIN, index_col=0)
    augmPatLvDiv_VALIDATION = pd.read_csv(augmPatLvDiv_VALIDATION, index_col=0)
    augmRndDiv_TRAIN = pd.read_csv(augmRndDiv_TRAIN, index_col=0)
    augmRndDiv_VALIDATION = pd.read_csv(augmRndDiv_VALIDATION, index_col=0)
    patLvDiv_TEST = pd.read_csv(patLvDiv_TEST, index_col=0)
    patLvDiv_TRAIN = pd.read_csv(patLvDiv_TRAIN, index_col=0)
    patLvDiv_VALIDATION = pd.read_csv(patLvDiv_VALIDATION, index_col=0)
    rndDiv_TEST = pd.read_csv(rndDiv_TEST, index_col=0)
    rndDiv_TRAIN = pd.read_csv(rndDiv_TRAIN, index_col=0)
    rndDiv_VALIDATION = pd.read_csv(rndDiv_VALIDATION, index_col=0)

    rndDiv_TRAIN = balanceDataset(rndDiv_TRAIN)
    rndDiv_VALIDATION = balanceDataset(rndDiv_VALIDATION)
    patLvDiv_TRAIN = balanceDataset(patLvDiv_TRAIN)
    patLvDiv_VALIDATION = balanceDataset(patLvDiv_VALIDATION)
    
    #testLSVM()
    #printTable(Path('results/L-SVM_TestPerformance_ContourData.csv'))

    #testQSVM()
    #printTable(Path('results/Q-SVM_TestPerformance_ContourData.csv'))
    
    #testPSVM()
    #printTable(Path('results/P-SVM_TestPerformance_ContourData.csv'))

    #testRSVM()
    #printTable(Path('results/R-SVM_TestPerformance_ContourData.csv'))
    
    #testSigSVM()
    #printTable(Path('results/Sig-SVM_TestPerformance_ContourData.csv'))
    
    ###############################################################################
    
    #testKNN()
    #printTable(Path('results/KNN_TestPerformance_ContourData.csv'))
    
    ###############################################################################
    
    #testRF()
    #printTable(Path('results/RF_TestPerformance_ContourData.csv'))
    
    ###############################################################################

    #testNN()

    ###############################################################################

    #t0 = balanceDataset(patLvDiv_TEST)
    #t1 = balanceDataset(rndDiv_TEST)
    #print(Counter(t0['cellType(ALL=1, HEM=-1)']))
    #print(Counter(t1['cellType(ALL=1, HEM=-1)']))

    print(f"\nEnd Script!\n{'#'*50}")
