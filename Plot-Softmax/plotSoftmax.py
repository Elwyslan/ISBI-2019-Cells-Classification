import numpy as np
from sklearn.decomposition import PCA
from sklearn import svm, neighbors, ensemble
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot as plt
from scipy.spatial import distance
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('ggplot')

if __name__ == '__main__':
    SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    x = ['C1'.translate(SUB), 'C2'.translate(SUB), 'C3'.translate(SUB), 'C4'.translate(SUB)]
    fig = plt.figure(figsize=(15, 8))

    ax = fig.add_subplot(121)
    y = [0,1,0,0]
    plt.bar(x,y,width=0.3,color='blue')
    ax.set_ylim([0.0, 1.05])
    ax.set_title('Distribuição de probabilidade esperada (p)', **{'fontname':'Times New Roman','size': 20})

    ax = fig.add_subplot(122)
    y = [0.075, 0.747, 0.15, 0.028]
    plt.bar(x,y,width=0.3,color='blue')
    ax.set_ylim([0.0, 1.05])
    ax.set_title('Distribuição de probabilidade inferida pela rede (q)', **{'fontname':'Times New Roman','size': 20})
    

    plt.savefig('distribuicoes-p-q.png')
    plt.show()

    print(f"\nEnd Script!\n{'#'*50}")





















