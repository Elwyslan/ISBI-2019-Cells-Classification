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
    x = np.arange(-10.0, 10.0, 0.01)
    y = 1.0/(1+np.exp(-x))
    
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111)
    ax.plot(x, y, linewidth=5.0, c='blue')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    
    plt.savefig('sigmoidFunc.png')
    plt.show()

    ############################################################
    ############################################################
    ############################################################

    x = np.arange(-10.0, 10.0, 0.01)
    y = np.tanh(x)
    
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111)
    ax.plot(x, y, linewidth=5.0, c='blue')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    
    plt.savefig('tanhFunc.png')
    plt.show()

    ############################################################
    ############################################################
    ############################################################

    x = np.arange(-10.0, 10.0, 0.01)
    y = np.maximum(x, 0)
    
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111)
    ax.plot(x, y, linewidth=5.0, c='blue')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    
    plt.savefig('reluFunc.png')
    plt.show()

    print(f"\nEnd Script!\n{'#'*50}")





















