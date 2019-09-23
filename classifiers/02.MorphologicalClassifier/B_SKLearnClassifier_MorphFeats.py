from pathlib import Path
import multiprocessing
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import svm, neighbors, ensemble
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import time

augmPatLvDiv_TRAIN = Path('data/AugmPatLvDiv_TRAIN-MORPHData_20-Features_20000-images.csv')
augmPatLvDiv_VALIDATION = Path('data/AugmPatLvDiv_VALIDATION-MORPHData_20-Features_5000-images.csv')
augmRndDiv_TRAIN = Path('data/AugmRndDiv_TRAIN-MORPHData_20-Features_20000-images.csv')
augmRndDiv_VALIDATION = Path('data/AugmRndDiv_VALIDATION-MORPHData_20-Features_5000-images.csv')
patLvDiv_TEST = Path('data/PatLvDiv_TEST-MORPHData_20-Features_550-images.csv')
patLvDiv_TRAIN = Path('data/PatLvDiv_TRAIN-MORPHData_20-Features_7224-images.csv')
patLvDiv_VALIDATION = Path('data/PatLvDiv_VALIDATION-MORPHData_20-Features_2887-images.csv')
rndDiv_TEST = Path('data/rndDiv_TEST-MORPHData_20-Features_1066-images.csv')
rndDiv_TRAIN = Path('data/rndDiv_TRAIN-MORPHData_20-Features_6398-images.csv')
rndDiv_VALIDATION = Path('data/rndDiv_VALIDATION-MORPHData_20-Features_3197-images.csv')

def balanceDataset(df):
    targets = df['cellType(ALL=1, HEM=-1)']
    cnt = dict(Counter(targets))
    if cnt[-1.0] == cnt[1.0]:
        return df
    ALLtargets = targets.where(targets==1)
    ALLtargets.dropna(axis=0, how='any', inplace=True)
    ALLtargets = ALLtargets.index.tolist()
    HEMtargets = targets.where(targets==-1)
    HEMtargets.dropna(axis=0, how='any', inplace=True)
    HEMtargets = HEMtargets.index.tolist()
    np.random.shuffle(ALLtargets)
    np.random.shuffle(HEMtargets)
    if len(ALLtargets) > len(HEMtargets):
        df = df.drop(ALLtargets[0:(len(ALLtargets) - len(HEMtargets))])
    elif len(HEMtargets) > len(ALLtargets):
        df = df.drop(HEMtargets[0:(len(HEMtargets) - len(ALLtargets))])
    return df

def computeStats(tn, fp, fn, tp):
    #Epsilon avoid division by zero
    epsilon = 1e-9
    #sensitivity/recall =  measures the proportion of actual positives that are correctly identified as such
    sensitivity = tp/(tp+fn+epsilon)
    #specificity = measures the proportion of actual negatives that are correctly identified as such
    specificity = tn/(tn+fp+epsilon)
    #accuracy = measures the systematic error
    accuracy = (tp+tn)/(tp+tn+fp+fn+epsilon)
    #precision = description of random errors, a measure of statistical variability.
    precision = tp/(tp+fp+epsilon)

    sensitivity = np.round((sensitivity*100.0), decimals=2)
    specificity = np.round((specificity*100.0), decimals=2)
    accuracy = np.round((accuracy*100.0), decimals=2)
    precision = np.round((precision*100.0), decimals=2)

    print('###################')
    print(f'Sensitivity: {sensitivity}%')
    print(f'Specificity: {specificity}%')
    print(f'Accuracy: {accuracy}%')
    print(f'Precision: {precision}%')
    print('###################')
    return sensitivity, specificity, accuracy, precision

def Xy_Split(df, normalizeData=None):
    y = df['cellType(ALL=1, HEM=-1)'].values
    X = df.drop(['cellType(ALL=1, HEM=-1)'], axis=1)
    #Normalize Features
    if normalizeData is not None:
        for col in X.columns:
            X[col] = (X[col] - normalizeData[col].mean()) / normalizeData[col].std() #mean=0, std=1
    else:
        for col in X.columns:
            X[col] = (X[col] - X[col].mean()) / X[col].std() #mean=0, std=1
    X = X.values
    return X, y

def validateCLF(train_df, valid_df, clf, PCA_matrix=None):
    X_valid, y_valid = Xy_Split(valid_df, normalizeData=train_df)
    if PCA_matrix is not None:
        print(f'Reduce PCA: {X_valid.shape[1]} to {PCA_matrix.n_components_}')
        X_valid = PCA_matrix.transform(X_valid)

    #Predict
    y_pred = []
    y_true = list(y_valid)
    for i in range(len(X_valid)):
        pred = clf.predict(X_valid[i].reshape(1,-1))[0]
        if pred==1:
            y_pred.append('ALL')
        elif pred==-1:
            y_pred.append('HEM')

        if y_true[i]==1:
            y_true[i] = 'ALL'
        elif y_true[i]==-1:
            y_true[i] = 'HEM'
        #print(f'{i} => True:{y_true[i]} - Predict:{y_pred[i]}')
    return y_true, y_pred

###############################################################################
def L_SVM_CLF(train_df, valid_df, params, PCA_Feats=None):
    X, y = Xy_Split(train_df, normalizeData=None)
    pca=None
    if isinstance(PCA_Feats, int):
        print(f'Reduce PCA: {X.shape[1]} to {PCA_Feats}')
        pca = PCA(n_components=PCA_Feats)
        pca.fit(X)
        X = pca.transform(X)
    #Linear Support Vector Classification
    linear_svc = svm.SVC(C=params['C'],
                         kernel='linear',
                         shrinking=True,
                         tol=params['tol'],
                         cache_size=200,
                         verbose=True,
                         max_iter=params['max_iter'])
    print('----------- Trainning -----------')
    linear_svc.fit(X, y)
    print('\n----------- Validation -----------')
    return validateCLF(train_df, valid_df, linear_svc, PCA_matrix=pca)
###############################################################################

def Q_SVM_CLF(train_df, valid_df, params, PCA_Feats=None):
    X, y = Xy_Split(train_df, normalizeData=None)
    pca=None
    if isinstance(PCA_Feats, int):
        print(f'Reduce PCA: {X.shape[1]} to {PCA_Feats}')
        pca = PCA(n_components=PCA_Feats)
        pca.fit(X)
        X = pca.transform(X)
    #Quadratic Support Vector Classification
    quadratic_svc = svm.SVC(C=params['C'],
                            kernel='poly',
                            degree=2,
                            gamma='auto',
                            coef0=params['coef0'],
                            shrinking=True,
                            tol=params['tol'],
                            cache_size=200,
                            verbose=True,
                            max_iter=params['max_iter'])
    print('----------- Trainning -----------')
    quadratic_svc.fit(X, y)
    print('\n----------- Validation -----------')
    return validateCLF(train_df, valid_df, quadratic_svc, PCA_matrix=pca)
###############################################################################

def Poly_SVM_CLF(train_df, valid_df, params, PCA_Feats=None):
    X, y = Xy_Split(train_df, normalizeData=None)
    pca=None
    if isinstance(PCA_Feats, int):
        print(f'Reduce PCA: {X.shape[1]} to {PCA_Feats}')
        pca = PCA(n_components=PCA_Feats)
        pca.fit(X)
        X = pca.transform(X)
    #Polynomial Support Vector Classification
    poly_svc = svm.SVC(C=params['C'],
                       kernel='poly',
                       degree=3,
                       gamma='auto',
                       coef0=params['coef0'],
                       shrinking=True,
                       tol=params['tol'],
                       cache_size=200,
                       verbose=True,
                       max_iter=params['max_iter'])
    print('----------- Trainning -----------')
    poly_svc.fit(X, y)
    print('\n----------- Validation -----------')
    return validateCLF(train_df, valid_df, poly_svc, PCA_matrix=pca)

def RBF_SVM_CLF(train_df, valid_df, params, PCA_Feats=None):
    X, y = Xy_Split(train_df, normalizeData=None)
    pca=None
    if isinstance(PCA_Feats, int):
        print(f'Reduce PCA: {X.shape[1]} to {PCA_Feats}')
        pca = PCA(n_components=PCA_Feats)
        pca.fit(X)
        X = pca.transform(X)

    #Linear Support Vector Classification
    rbf_svc = svm.SVC(C=params['C'],
                      kernel='rbf',
                      gamma='auto',
                      shrinking=True,
                      tol=params['tol'],
                      cache_size=200,
                      verbose=True,
                      max_iter=params['max_iter'])
    print('----------- Trainning -----------')
    rbf_svc.fit(X, y)
    print('\n----------- Validation -----------')
    return validateCLF(train_df, valid_df, rbf_svc, PCA_matrix=pca)

def Sigm_SVM_CLF(train_df, valid_df, params={'C':1.0, 'coef0':0.0,'tol':0.001, 'max_iter':-1}, PCA_Feats=None):
    X, y = Xy_Split(train_df, normalizeData=None)
    pca=None
    if isinstance(PCA_Feats, int):
        print(f'Reduce PCA: {X.shape[1]} to {PCA_Feats}')
        pca = PCA(n_components=PCA_Feats)
        pca.fit(X)
        X = pca.transform(X)
    #Linear Support Vector Classification
    sigm_svc = svm.SVC(C=params['C'],
                         kernel='sigmoid',
                         gamma='auto',
                         coef0=params['coef0'],
                         shrinking=True,
                         tol=params['tol'],
                         cache_size=200,
                         verbose=True,
                         max_iter=params['max_iter'])
    print('----------- Trainning -----------')
    sigm_svc.fit(X, y)
    print('\n----------- Validation -----------')
    return validateCLF(train_df, valid_df, sigm_svc, PCA_matrix=pca)

def KNN_CLF(train_df, valid_df, n_neighbors, PCA_Feats=None):
    X, y = Xy_Split(train_df, normalizeData=None)
    pca=None
    if isinstance(PCA_Feats, int):
        print(f'Reduce PCA: {X.shape[1]} to {PCA_Feats}')
        pca = PCA(n_components=PCA_Feats)
        pca.fit(X)
        X = pca.transform(X)
    #K-Neighbors Classification
    knn = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    print('----------- Trainning -----------')
    knn.fit(X, y)
    print('\n----------- Validation -----------')
    return validateCLF(train_df, valid_df, knn, PCA_matrix=pca)

def RF_CLF(train_df, valid_df, n_estimators, PCA_Feats=None):
    X, y = Xy_Split(train_df, normalizeData=None)
    pca=None
    if isinstance(PCA_Feats, int):
        print(f'Reduce PCA: {X.shape[1]} to {PCA_Feats}')
        pca = PCA(n_components=PCA_Feats)
        pca.fit(X)
        X = pca.transform(X)
    #Random Forest Classification
    rf = ensemble.RandomForestClassifier(n_estimators=n_estimators)
    print('----------- Trainning -----------')
    rf.fit(X, y)
    print('\n----------- Validation -----------')
    return validateCLF(train_df, valid_df, rf, PCA_matrix=pca)












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

    pcaNofFeat = [i for i in range(5,25,5)]
    
    ###############################################################################
    def LSVMaugmPatLvDiv():
        df = pd.DataFrame()
        for feat in pcaNofFeat:
            collectedData = {}
            y_true, y_pred = L_SVM_CLF(augmPatLvDiv_TRAIN,
                                       augmPatLvDiv_VALIDATION,
                                       params={'C':1.0, 'tol':0.001, 'max_iter':-1},
                                       PCA_Feats=feat)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=['HEM', 'ALL']).ravel()
            sens, spec, acc, prec = computeStats(tn, fp, fn, tp)
            collectedData[f'Features'] = feat
            collectedData[f'L-SVM_sens'] = sens
            collectedData[f'L-SVM_spec'] = spec
            collectedData[f'L-SVM_acc'] = acc
            collectedData[f'L-SVM_prec'] = prec
            df = df.append(collectedData, ignore_index=True)
        df.to_csv(f'results/L-SVM_ValidPerformance_MorphologicalData_AugmPatLvDivImages.csv')
    ###############################################################################
    def LSVMaugmRndDiv():
        df = pd.DataFrame()
        for feat in pcaNofFeat:
            collectedData = {}
            y_true, y_pred = L_SVM_CLF(augmRndDiv_TRAIN,
                                       augmRndDiv_VALIDATION,
                                       params={'C':1.0, 'tol':0.001, 'max_iter':-1},
                                       PCA_Feats=feat)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=['HEM', 'ALL']).ravel()
            sens, spec, acc, prec = computeStats(tn, fp, fn, tp)
            collectedData[f'Features'] = feat
            collectedData[f'L-SVM_sens'] = sens
            collectedData[f'L-SVM_spec'] = spec
            collectedData[f'L-SVM_acc'] = acc
            collectedData[f'L-SVM_prec'] = prec
            df = df.append(collectedData, ignore_index=True)
        df.to_csv(f'results/L-SVM_ValidPerformance_MorphologicalData_AugmRndDivImages.csv')
    ###############################################################################
    def LSVMpatLvDiv():
        df = pd.DataFrame()
        for feat in pcaNofFeat:
            collectedData = {}
            y_true, y_pred = L_SVM_CLF(patLvDiv_TRAIN,
                                       patLvDiv_VALIDATION,
                                       params={'C':1.0, 'tol':0.001, 'max_iter':-1},
                                       PCA_Feats=feat)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=['HEM', 'ALL']).ravel()
            sens, spec, acc, prec = computeStats(tn, fp, fn, tp)
            collectedData[f'Features'] = feat
            collectedData[f'L-SVM_sens'] = sens
            collectedData[f'L-SVM_spec'] = spec
            collectedData[f'L-SVM_acc'] = acc
            collectedData[f'L-SVM_prec'] = prec
            df = df.append(collectedData, ignore_index=True)
        df.to_csv(f'results/L-SVM_ValidPerformance_MorphologicalData_PatLvDivImages.csv')
    ###############################################################################
    def LSVMrndDiv():
        df = pd.DataFrame()
        for feat in pcaNofFeat:
            collectedData = {}
            y_true, y_pred = L_SVM_CLF(rndDiv_TRAIN,
                                       rndDiv_VALIDATION,
                                       params={'C':1.0, 'tol':0.001, 'max_iter':-1},
                                       PCA_Feats=feat)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=['HEM', 'ALL']).ravel()
            sens, spec, acc, prec = computeStats(tn, fp, fn, tp)
            collectedData[f'Features'] = feat
            collectedData[f'L-SVM_sens'] = sens
            collectedData[f'L-SVM_spec'] = spec
            collectedData[f'L-SVM_acc'] = acc
            collectedData[f'L-SVM_prec'] = prec
            df = df.append(collectedData, ignore_index=True)
        df.to_csv(f'results/L-SVM_ValidPerformance_MorphologicalData_RndDivImages.csv')
    ###############################################################################
    p0 = multiprocessing.Process(name='LSVMaugmPatLvDiv', target=LSVMaugmPatLvDiv)
    p1 = multiprocessing.Process(name='LSVMaugmRndDiv',target=LSVMaugmRndDiv)
    p2 = multiprocessing.Process(name='LSVMpatLvDiv',target=LSVMpatLvDiv)
    p3 = multiprocessing.Process(name='LSVMrndDiv',target=LSVMrndDiv)

    ###############################################################################
    #*****************************************************************************#
    #*****************************************************************************#
    #*****************************************************************************#
    ###############################################################################

    def QSVMaugmPatLvDiv():
        df = pd.DataFrame()
        for feat in pcaNofFeat:
            collectedData = {}
            y_true, y_pred = Q_SVM_CLF(augmPatLvDiv_TRAIN,
                                       augmPatLvDiv_VALIDATION,
                                       params={'C':1.0, 'coef0':0.0,'tol':0.001, 'max_iter':-1},
                                       PCA_Feats=feat)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=['HEM', 'ALL']).ravel()
            sens, spec, acc, prec = computeStats(tn, fp, fn, tp)
            collectedData[f'Features'] = feat
            collectedData[f'Q-SVM_sens'] = sens
            collectedData[f'Q-SVM_spec'] = spec
            collectedData[f'Q-SVM_acc'] = acc
            collectedData[f'Q-SVM_prec'] = prec
            df = df.append(collectedData, ignore_index=True)
        df.to_csv(f'results/Q-SVM_ValidPerformance_MorphologicalData_AugmPatLvDivImages.csv')
    ###############################################################################
    def QSVMaugmRndDiv():
        df = pd.DataFrame()
        for feat in pcaNofFeat:
            collectedData = {}
            y_true, y_pred = Q_SVM_CLF(augmRndDiv_TRAIN,
                                       augmRndDiv_VALIDATION,
                                       params={'C':1.0, 'coef0':0.0,'tol':0.001, 'max_iter':-1},
                                       PCA_Feats=feat)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=['HEM', 'ALL']).ravel()
            sens, spec, acc, prec = computeStats(tn, fp, fn, tp)
            collectedData[f'Features'] = feat
            collectedData[f'Q-SVM_sens'] = sens
            collectedData[f'Q-SVM_spec'] = spec
            collectedData[f'Q-SVM_acc'] = acc
            collectedData[f'Q-SVM_prec'] = prec
            df = df.append(collectedData, ignore_index=True)
        df.to_csv(f'results/Q-SVM_ValidPerformance_MorphologicalData_AugmRndDivImages.csv')
    ###############################################################################
    def QSVMpatLvDiv():
        df = pd.DataFrame()
        for feat in pcaNofFeat:
            collectedData = {}
            y_true, y_pred = Q_SVM_CLF(patLvDiv_TRAIN,
                                       patLvDiv_VALIDATION,
                                       params={'C':1.0, 'coef0':0.0,'tol':0.001, 'max_iter':-1},
                                       PCA_Feats=feat)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=['HEM', 'ALL']).ravel()
            sens, spec, acc, prec = computeStats(tn, fp, fn, tp)
            collectedData[f'Features'] = feat
            collectedData[f'Q-SVM_sens'] = sens
            collectedData[f'Q-SVM_spec'] = spec
            collectedData[f'Q-SVM_acc'] = acc
            collectedData[f'Q-SVM_prec'] = prec
            df = df.append(collectedData, ignore_index=True)
        df.to_csv(f'results/Q-SVM_ValidPerformance_MorphologicalData_PatLvDivImages.csv')
    ###############################################################################
    def QSVMrndDiv():
        df = pd.DataFrame()
        for feat in pcaNofFeat:
            collectedData = {}
            y_true, y_pred = Q_SVM_CLF(rndDiv_TRAIN,
                                       rndDiv_VALIDATION,
                                       params={'C':1.0, 'coef0':0.0,'tol':0.001, 'max_iter':-1},
                                       PCA_Feats=feat)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=['HEM', 'ALL']).ravel()
            sens, spec, acc, prec = computeStats(tn, fp, fn, tp)
            collectedData[f'Features'] = feat
            collectedData[f'Q-SVM_sens'] = sens
            collectedData[f'Q-SVM_spec'] = spec
            collectedData[f'Q-SVM_acc'] = acc
            collectedData[f'Q-SVM_prec'] = prec
            df = df.append(collectedData, ignore_index=True)
        df.to_csv(f'results/Q-SVM_ValidPerformance_MorphologicalData_RndDivImages.csv')
    ###############################################################################
    p4 = multiprocessing.Process(name='QSVMaugmPatLvDiv', target=QSVMaugmPatLvDiv)
    p5 = multiprocessing.Process(name='QSVMaugmRndDiv',target=QSVMaugmRndDiv)
    p6 = multiprocessing.Process(name='QSVMpatLvDiv',target=QSVMpatLvDiv)
    p7 = multiprocessing.Process(name='QSVMrndDiv',target=QSVMrndDiv)

    ###############################################################################
    #*****************************************************************************#
    #*****************************************************************************#
    #*****************************************************************************#
    ###############################################################################
    
    def PolySVMaugmPatLvDiv():
        df = pd.DataFrame()
        for feat in pcaNofFeat:
            collectedData = {}
            y_true, y_pred = Poly_SVM_CLF(augmPatLvDiv_TRAIN,
                                          augmPatLvDiv_VALIDATION,
                                          params={'C':1.0, 'coef0':0.0,'tol':0.001, 'max_iter':1e6},
                                          PCA_Feats=feat)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=['HEM', 'ALL']).ravel()
            sens, spec, acc, prec = computeStats(tn, fp, fn, tp)
            collectedData[f'Features'] = feat
            collectedData[f'P-SVM_sens'] = sens
            collectedData[f'P-SVM_spec'] = spec
            collectedData[f'P-SVM_acc'] = acc
            collectedData[f'P-SVM_prec'] = prec
            df = df.append(collectedData, ignore_index=True)
        df.to_csv(f'results/P-SVM_ValidPerformance_MorphologicalData_AugmPatLvDivImages.csv')
    ###############################################################################
    def PolySVMaugmRndDiv():
        df = pd.DataFrame()
        for feat in pcaNofFeat:
            collectedData = {}
            y_true, y_pred = Poly_SVM_CLF(augmRndDiv_TRAIN,
                                          augmRndDiv_VALIDATION,
                                          params={'C':1.0, 'coef0':0.0,'tol':0.001, 'max_iter':1e6},
                                          PCA_Feats=feat)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=['HEM', 'ALL']).ravel()
            sens, spec, acc, prec = computeStats(tn, fp, fn, tp)
            collectedData[f'Features'] = feat
            collectedData[f'P-SVM_sens'] = sens
            collectedData[f'P-SVM_spec'] = spec
            collectedData[f'P-SVM_acc'] = acc
            collectedData[f'P-SVM_prec'] = prec
            df = df.append(collectedData, ignore_index=True)
        df.to_csv(f'results/P-SVM_ValidPerformance_MorphologicalData_AugmRndDivImages.csv')
    ###############################################################################
    def PolySVMpatLvDiv():
        df = pd.DataFrame()
        for feat in pcaNofFeat:
            collectedData = {}
            y_true, y_pred = Poly_SVM_CLF(patLvDiv_TRAIN,
                                          patLvDiv_VALIDATION,
                                          params={'C':1.0, 'coef0':0.0,'tol':0.001, 'max_iter':1e6},
                                          PCA_Feats=feat)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=['HEM', 'ALL']).ravel()
            sens, spec, acc, prec = computeStats(tn, fp, fn, tp)
            collectedData[f'Features'] = feat
            collectedData[f'P-SVM_sens'] = sens
            collectedData[f'P-SVM_spec'] = spec
            collectedData[f'P-SVM_acc'] = acc
            collectedData[f'P-SVM_prec'] = prec
            df = df.append(collectedData, ignore_index=True)
        df.to_csv(f'results/P-SVM_ValidPerformance_MorphologicalData_PatLvDivImages.csv')
    ###############################################################################
    def PolySVMrndDiv():
        df = pd.DataFrame()
        for feat in pcaNofFeat:
            collectedData = {}
            y_true, y_pred = Poly_SVM_CLF(rndDiv_TRAIN,
                                          rndDiv_VALIDATION,
                                          params={'C':1.0, 'coef0':0.0,'tol':0.001, 'max_iter':1e6},
                                          PCA_Feats=feat)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=['HEM', 'ALL']).ravel()
            sens, spec, acc, prec = computeStats(tn, fp, fn, tp)
            collectedData[f'Features'] = feat
            collectedData[f'P-SVM_sens'] = sens
            collectedData[f'P-SVM_spec'] = spec
            collectedData[f'P-SVM_acc'] = acc
            collectedData[f'P-SVM_prec'] = prec
            df = df.append(collectedData, ignore_index=True)
        df.to_csv(f'results/P-SVM_ValidPerformance_MorphologicalData_RndDivImages.csv')
    ###############################################################################
    p8 = multiprocessing.Process(name='PolySVMaugmPatLvDiv', target=PolySVMaugmPatLvDiv)
    p9 = multiprocessing.Process(name='PolySVMaugmRndDiv',target=PolySVMaugmRndDiv)
    p10 = multiprocessing.Process(name='PolySVMpatLvDiv',target=PolySVMpatLvDiv)
    p11 = multiprocessing.Process(name='PolySVMrndDiv',target=PolySVMrndDiv)
    
    ###############################################################################
    #*****************************************************************************#
    #*****************************************************************************#
    #*****************************************************************************#
    ###############################################################################
    
    def RBFSVMaugmPatLvDiv():
        df = pd.DataFrame()
        for feat in pcaNofFeat:
            collectedData = {}
            y_true, y_pred = RBF_SVM_CLF(augmPatLvDiv_TRAIN,
                                         augmPatLvDiv_VALIDATION,
                                         params={'C':1.0,'tol':0.001, 'max_iter':-1},
                                         PCA_Feats=feat)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=['HEM', 'ALL']).ravel()
            sens, spec, acc, prec = computeStats(tn, fp, fn, tp)
            collectedData[f'Features'] = feat
            collectedData[f'R-SVM_sens'] = sens
            collectedData[f'R-SVM_spec'] = spec
            collectedData[f'R-SVM_acc'] = acc
            collectedData[f'R-SVM_prec'] = prec
            df = df.append(collectedData, ignore_index=True)
        df.to_csv(f'results/R-SVM_ValidPerformance_MorphologicalData_AugmPatLvDivImages.csv')
    ###############################################################################
    def RBFSVMaugmRndDiv():
        df = pd.DataFrame()
        for feat in pcaNofFeat:
            collectedData = {}
            y_true, y_pred = RBF_SVM_CLF(augmRndDiv_TRAIN,
                                         augmRndDiv_VALIDATION,
                                         params={'C':1.0,'tol':0.001, 'max_iter':-1},
                                         PCA_Feats=feat)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=['HEM', 'ALL']).ravel()
            sens, spec, acc, prec = computeStats(tn, fp, fn, tp)
            collectedData[f'Features'] = feat
            collectedData[f'R-SVM_sens'] = sens
            collectedData[f'R-SVM_spec'] = spec
            collectedData[f'R-SVM_acc'] = acc
            collectedData[f'R-SVM_prec'] = prec
            df = df.append(collectedData, ignore_index=True)
        df.to_csv(f'results/R-SVM_ValidPerformance_MorphologicalData_AugmRndDivImages.csv')
    ###############################################################################
    def RBFSVMpatLvDiv():
        df = pd.DataFrame()
        for feat in pcaNofFeat:
            collectedData = {}
            y_true, y_pred = RBF_SVM_CLF(patLvDiv_TRAIN,
                                         patLvDiv_VALIDATION,
                                         params={'C':1.0,'tol':0.001, 'max_iter':-1},
                                         PCA_Feats=feat)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=['HEM', 'ALL']).ravel()
            sens, spec, acc, prec = computeStats(tn, fp, fn, tp)
            collectedData[f'Features'] = feat
            collectedData[f'R-SVM_sens'] = sens
            collectedData[f'R-SVM_spec'] = spec
            collectedData[f'R-SVM_acc'] = acc
            collectedData[f'R-SVM_prec'] = prec
            df = df.append(collectedData, ignore_index=True)
        df.to_csv(f'results/R-SVM_ValidPerformance_MorphologicalData_PatLvDivImages.csv')
    ###############################################################################
    def RBFSVMrndDiv():
        df = pd.DataFrame()
        for feat in pcaNofFeat:
            collectedData = {}
            y_true, y_pred = RBF_SVM_CLF(rndDiv_TRAIN,
                                         rndDiv_VALIDATION,
                                         params={'C':1.0,'tol':0.001, 'max_iter':-1},
                                         PCA_Feats=feat)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=['HEM', 'ALL']).ravel()
            sens, spec, acc, prec = computeStats(tn, fp, fn, tp)
            collectedData[f'Features'] = feat
            collectedData[f'R-SVM_sens'] = sens
            collectedData[f'R-SVM_spec'] = spec
            collectedData[f'R-SVM_acc'] = acc
            collectedData[f'R-SVM_prec'] = prec
            df = df.append(collectedData, ignore_index=True)
        df.to_csv(f'results/R-SVM_ValidPerformance_MorphologicalData_RndDivImages.csv')
    ###############################################################################
    p12 = multiprocessing.Process(name='RBFSVMaugmPatLvDiv', target=RBFSVMaugmPatLvDiv)
    p13 = multiprocessing.Process(name='RBFSVMaugmRndDiv',target=RBFSVMaugmRndDiv)
    p14 = multiprocessing.Process(name='RBFSVMpatLvDiv',target=RBFSVMpatLvDiv)
    p15 = multiprocessing.Process(name='RBFSVMrndDiv',target=RBFSVMrndDiv)

    ###############################################################################
    #*****************************************************************************#
    #*****************************************************************************#
    #*****************************************************************************#
    ###############################################################################

    def SigmoidSVMaugmPatLvDiv():
        df = pd.DataFrame()
        for feat in pcaNofFeat:
            collectedData = {}
            y_true, y_pred = Sigm_SVM_CLF(augmPatLvDiv_TRAIN,
                                          augmPatLvDiv_VALIDATION,
                                          params={'C':1.0, 'coef0':0.0,'tol':0.001, 'max_iter':-1},
                                          PCA_Feats=feat)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=['HEM', 'ALL']).ravel()
            sens, spec, acc, prec = computeStats(tn, fp, fn, tp)
            collectedData[f'Features'] = feat
            collectedData[f'S-SVM_sens'] = sens
            collectedData[f'S-SVM_spec'] = spec
            collectedData[f'S-SVM_acc'] = acc
            collectedData[f'S-SVM_prec'] = prec
            df = df.append(collectedData, ignore_index=True)
        df.to_csv(f'results/Sig-SVM_ValidPerformance_MorphologicalData_AugmPatLvDivImages.csv')
    ###############################################################################
    def SigmoidSVMaugmRndDiv():
        df = pd.DataFrame()
        for feat in pcaNofFeat:
            collectedData = {}
            y_true, y_pred = Sigm_SVM_CLF(augmRndDiv_TRAIN,
                                          augmRndDiv_VALIDATION,
                                          params={'C':1.0, 'coef0':0.0,'tol':0.001, 'max_iter':-1},
                                          PCA_Feats=feat)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=['HEM', 'ALL']).ravel()
            sens, spec, acc, prec = computeStats(tn, fp, fn, tp)
            collectedData[f'Features'] = feat
            collectedData[f'S-SVM_sens'] = sens
            collectedData[f'S-SVM_spec'] = spec
            collectedData[f'S-SVM_acc'] = acc
            collectedData[f'S-SVM_prec'] = prec
            df = df.append(collectedData, ignore_index=True)
        df.to_csv(f'results/Sig-SVM_ValidPerformance_MorphologicalData_AugmRndDivImages.csv')
    ###############################################################################
    def SigmoidSVMpatLvDiv():
        df = pd.DataFrame()
        for feat in pcaNofFeat:
            collectedData = {}
            y_true, y_pred = Sigm_SVM_CLF(patLvDiv_TRAIN,
                                          patLvDiv_VALIDATION,
                                          params={'C':1.0, 'coef0':0.0,'tol':0.001, 'max_iter':-1},
                                          PCA_Feats=feat)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=['HEM', 'ALL']).ravel()
            sens, spec, acc, prec = computeStats(tn, fp, fn, tp)
            collectedData[f'Features'] = feat
            collectedData[f'S-SVM_sens'] = sens
            collectedData[f'S-SVM_spec'] = spec
            collectedData[f'S-SVM_acc'] = acc
            collectedData[f'S-SVM_prec'] = prec
            df = df.append(collectedData, ignore_index=True)
        df.to_csv(f'results/Sig-SVM_ValidPerformance_MorphologicalData_PatLvDivImages.csv')
    ###############################################################################
    def SigmoidSVMrndDiv():
        df = pd.DataFrame()
        for feat in pcaNofFeat:
            collectedData = {}
            y_true, y_pred = Sigm_SVM_CLF(rndDiv_TRAIN,
                                          rndDiv_VALIDATION,
                                          params={'C':1.0, 'coef0':0.0,'tol':0.001, 'max_iter':-1},
                                          PCA_Feats=feat)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=['HEM', 'ALL']).ravel()
            sens, spec, acc, prec = computeStats(tn, fp, fn, tp)
            collectedData[f'Features'] = feat
            collectedData[f'S-SVM_sens'] = sens
            collectedData[f'S-SVM_spec'] = spec
            collectedData[f'S-SVM_acc'] = acc
            collectedData[f'S-SVM_prec'] = prec
            df = df.append(collectedData, ignore_index=True)
        df.to_csv(f'results/Sig-SVM_ValidPerformance_MorphologicalData_RndDivImages.csv')
    ###############################################################################
    p16 = multiprocessing.Process(name='SigmoidSVMaugmPatLvDiv', target=SigmoidSVMaugmPatLvDiv)
    p17 = multiprocessing.Process(name='SigmoidSVMaugmRndDiv',target=SigmoidSVMaugmRndDiv)
    p18 = multiprocessing.Process(name='SigmoidSVMpatLvDiv',target=SigmoidSVMpatLvDiv)
    p19 = multiprocessing.Process(name='SigmoidSVMrndDiv',target=SigmoidSVMrndDiv)

    ###############################################################################
    #*****************************************************************************#
    #*****************************************************************************#
    #*****************************************************************************#
    ###############################################################################

    ###############################################################################
    def KNNaugmPatLvDiv():
        df = pd.DataFrame()
        for feat in pcaNofFeat:
            collectedData = {}
            collectedData[f'Features'] = feat
            for k in range(5,21):
                y_true, y_pred = KNN_CLF(augmPatLvDiv_TRAIN,
                                         augmPatLvDiv_VALIDATION,
                                         n_neighbors=k, PCA_Feats=feat)
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=['HEM', 'ALL']).ravel()
                sens, spec, acc, prec = computeStats(tn, fp, fn, tp)
                collectedData[f'{k}-KNN_sens'] = sens
                collectedData[f'{k}-KNN_spec'] = spec
                collectedData[f'{k}-KNN_acc'] = acc
                collectedData[f'{k}-KNN_prec'] = prec
            df = df.append(collectedData, ignore_index=True)
        df.to_csv(f'results/KNN_ValidPerformance_MorphologicalData_AugmPatLvDivImages.csv')
    ###############################################################################
    def KNNaugmRndDiv():
        df = pd.DataFrame()
        for feat in pcaNofFeat:
            collectedData = {}
            collectedData[f'Features'] = feat
            for k in range(5,21):
                y_true, y_pred = KNN_CLF(augmRndDiv_TRAIN,
                                         augmRndDiv_VALIDATION,
                                         n_neighbors=k, PCA_Feats=feat)
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=['HEM', 'ALL']).ravel()
                sens, spec, acc, prec = computeStats(tn, fp, fn, tp)
                collectedData[f'{k}-KNN_sens'] = sens
                collectedData[f'{k}-KNN_spec'] = spec
                collectedData[f'{k}-KNN_acc'] = acc
                collectedData[f'{k}-KNN_prec'] = prec
            df = df.append(collectedData, ignore_index=True)
        df.to_csv(f'results/KNN_ValidPerformance_MorphologicalData_AugmRndDivImages.csv')
    ###############################################################################
    def KNNpatLvDiv():
        df = pd.DataFrame()
        for feat in pcaNofFeat:
            collectedData = {}
            collectedData[f'Features'] = feat
            for k in range(5,21):
                y_true, y_pred = KNN_CLF(patLvDiv_TRAIN,
                                         patLvDiv_VALIDATION,
                                         n_neighbors=k, PCA_Feats=feat)
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=['HEM', 'ALL']).ravel()
                sens, spec, acc, prec = computeStats(tn, fp, fn, tp)
                collectedData[f'{k}-KNN_sens'] = sens
                collectedData[f'{k}-KNN_spec'] = spec
                collectedData[f'{k}-KNN_acc'] = acc
                collectedData[f'{k}-KNN_prec'] = prec
            df = df.append(collectedData, ignore_index=True)
        df.to_csv(f'results/KNN_ValidPerformance_MorphologicalData_PatLvDivImages.csv')
    ###############################################################################
    def KNNrndDiv():
        df = pd.DataFrame()
        for feat in pcaNofFeat:
            collectedData = {}
            collectedData[f'Features'] = feat
            for k in range(5,21):
                y_true, y_pred = KNN_CLF(rndDiv_TRAIN,
                                         rndDiv_VALIDATION,
                                         n_neighbors=k, PCA_Feats=feat)
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=['HEM', 'ALL']).ravel()
                sens, spec, acc, prec = computeStats(tn, fp, fn, tp)
                collectedData[f'{k}-KNN_sens'] = sens
                collectedData[f'{k}-KNN_spec'] = spec
                collectedData[f'{k}-KNN_acc'] = acc
                collectedData[f'{k}-KNN_prec'] = prec
            df = df.append(collectedData, ignore_index=True)
        df.to_csv(f'results/KNN_ValidPerformance_MorphologicalData_rndDivImages.csv')
    ###############################################################################
    p20 = multiprocessing.Process(name='KNNaugmPatLvDiv', target=KNNaugmPatLvDiv)
    p21 = multiprocessing.Process(name='KNNaugmRndDiv',target=KNNaugmRndDiv)
    p22 = multiprocessing.Process(name='KNNpatLvDiv',target=KNNpatLvDiv)
    p23 = multiprocessing.Process(name='KNNrndDiv',target=KNNrndDiv)

    ###############################################################################
    #*****************************************************************************#
    #*****************************************************************************#
    #*****************************************************************************#
    ###############################################################################

    ###############################################################################
    def RFaugmPatLvDiv():
        df = pd.DataFrame()
        for feat in pcaNofFeat:
            collectedData = {}
            collectedData[f'Features'] = feat
            for nEstm in range(50,170,20):
                y_true, y_pred = RF_CLF(augmPatLvDiv_TRAIN,
                                        augmPatLvDiv_VALIDATION,
                                        n_estimators=nEstm, PCA_Feats=feat)
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=['HEM', 'ALL']).ravel()
                sens, spec, acc, prec = computeStats(tn, fp, fn, tp)
                collectedData[f'{nEstm}-RF_sens'] = sens
                collectedData[f'{nEstm}-RF_spec'] = spec
                collectedData[f'{nEstm}-RF_acc'] = acc
                collectedData[f'{nEstm}-RF_prec'] = prec
            df = df.append(collectedData, ignore_index=True)
        df.to_csv(f'results/RF_ValidPerformance_MorphologicalData_AugmPatLvDivImages.csv')
    ###############################################################################
    def RFaugmRndDiv():
        df = pd.DataFrame()
        for feat in pcaNofFeat:
            collectedData = {}
            collectedData[f'Features'] = feat
            for nEstm in range(50,170,20):
                y_true, y_pred = RF_CLF(augmRndDiv_TRAIN,
                                        augmRndDiv_VALIDATION,
                                        n_estimators=nEstm, PCA_Feats=feat)
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=['HEM', 'ALL']).ravel()
                sens, spec, acc, prec = computeStats(tn, fp, fn, tp)
                collectedData[f'{nEstm}-RF_sens'] = sens
                collectedData[f'{nEstm}-RF_spec'] = spec
                collectedData[f'{nEstm}-RF_acc'] = acc
                collectedData[f'{nEstm}-RF_prec'] = prec
            df = df.append(collectedData, ignore_index=True)
        df.to_csv(f'results/RF_ValidPerformance_MorphologicalData_AugmRndDivImages.csv')
    ###############################################################################
    def RFpatLvDiv():
        df = pd.DataFrame()
        for feat in pcaNofFeat:
            collectedData = {}
            collectedData[f'Features'] = feat
            for nEstm in range(50,170,20):
                y_true, y_pred = RF_CLF(patLvDiv_TRAIN,
                                        patLvDiv_VALIDATION,
                                        n_estimators=nEstm, PCA_Feats=feat)
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=['HEM', 'ALL']).ravel()
                sens, spec, acc, prec = computeStats(tn, fp, fn, tp)
                collectedData[f'{nEstm}-RF_sens'] = sens
                collectedData[f'{nEstm}-RF_spec'] = spec
                collectedData[f'{nEstm}-RF_acc'] = acc
                collectedData[f'{nEstm}-RF_prec'] = prec
            df = df.append(collectedData, ignore_index=True)
        df.to_csv(f'results/RF_ValidPerformance_MorphologicalData_PatLvDivImages.csv')
    ###############################################################################
    def RFrndDiv():
        df = pd.DataFrame()
        for feat in pcaNofFeat:
            collectedData = {}
            collectedData[f'Features'] = feat
            for nEstm in range(50,170,20):
                y_true, y_pred = RF_CLF(rndDiv_TRAIN,
                                        rndDiv_VALIDATION,
                                        n_estimators=nEstm, PCA_Feats=feat)
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=['HEM', 'ALL']).ravel()
                sens, spec, acc, prec = computeStats(tn, fp, fn, tp)
                collectedData[f'{nEstm}-RF_sens'] = sens
                collectedData[f'{nEstm}-RF_spec'] = spec
                collectedData[f'{nEstm}-RF_acc'] = acc
                collectedData[f'{nEstm}-RF_prec'] = prec
            df = df.append(collectedData, ignore_index=True)
        df.to_csv(f'results/RF_ValidPerformance_MorphologicalData_RndDivImages.csv')
    ###############################################################################
    p24 = multiprocessing.Process(name='RFaugmPatLvDiv', target=RFaugmPatLvDiv)
    p25 = multiprocessing.Process(name='RFaugmRndDiv',target=RFaugmRndDiv)
    p26 = multiprocessing.Process(name='RFpatLvDiv',target=RFpatLvDiv)
    p27 = multiprocessing.Process(name='RFrndDiv',target=RFrndDiv)

    PROCESS_LIST = [p0, p1, p2, p3, p4,
                    p5, p6, p7, p8, p9,
                    p10, p11, p12, p13,
                    p14, p15, p16, p17,
                    p18, p19, p20, p21,
                    p22, p23, p24, p25,
                    p26, p27]
    while len(PROCESS_LIST)>0:
        while len(multiprocessing.active_children())<4 and len(PROCESS_LIST)>0:
            p = PROCESS_LIST.pop(0)
            p.start()
            time.sleep(10)
        time.sleep(10)

    while len(multiprocessing.active_children())>0:
        time.sleep(10)
    
    print(f"\nEnd Script!\n{'#'*50}")
