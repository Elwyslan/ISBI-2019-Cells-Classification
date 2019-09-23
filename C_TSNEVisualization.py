from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
plt.style.use('ggplot')

FOF_DATA_0 = Path('classifiers/00.firstOrderFeaturesClassifier/data/rndDiv_TRAIN-FOFData_108-Features_6398-images.csv')
FOF_DATA_1 = Path('classifiers/00.firstOrderFeaturesClassifier/data/rndDiv_TEST-FOFData_108-Features_1066-images.csv')
FOF_DATA_2 = Path('classifiers/00.firstOrderFeaturesClassifier/data/rndDiv_VALIDATION-FOFData_108-Features_3197-images.csv')

if __name__ == '__main__':
    df = []
    df.append(pd.read_csv(FOF_DATA_0, index_col=0))
    df.append(pd.read_csv(FOF_DATA_1, index_col=0))
    df.append(pd.read_csv(FOF_DATA_2, index_col=0))
    df = pd.concat(df)

    targets = df['cellType(ALL=1, HEM=-1)'].values
    df.drop(['cellType(ALL=1, HEM=-1)'], axis=1, inplace=True)
 
    for col in df.columns:
        df[col] = (df[col] - df[col].mean()) / df[col].std() #mean=0, std=1

    tsne = TSNE(n_components=2, verbose=2, n_iter=2000, n_iter_without_progress=2000)

    X = tsne.fit_transform(df.values)

    plt.figure(figsize=(10, 6))
    plt.scatter(X[targets==1,0], X[targets==1,1], marker='x', c='red', label='Células Malignas')
    plt.scatter(X[targets==-1,0], X[targets==-1,1], marker='x', c='blue', label='Células Saudáveis')
    plt.xlabel('X no espaço T-SNE',fontsize=18)
    plt.ylabel('Y no espaço T-SNE',fontsize=18)
    plt.legend(loc='upper right',prop={'size': 14, 'weight':'bold'})
    plt.tight_layout()
    plt.savefig('FOF_TSNE-Reduction.png')
    plt.show()

    print(f"\nEnd Script!\n{'#'*50}")
