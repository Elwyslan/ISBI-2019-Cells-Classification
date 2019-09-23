import numpy as np
from sklearn.decomposition import PCA
from sklearn import svm, neighbors, ensemble
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot as plt
from scipy.spatial import distance
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('ggplot')

def exampleSVM():
    #Create a random dataset and labels
    X, y = make_blobs(n_samples=50,
                      centers=2, random_state=0, cluster_std=0.60)
    #Train a SVM model
    model = svm.SVC(kernel='linear', verbose=True, C=1e10)
    model.fit(X, y)
    #Retrieve hyperplane coefficients
    w = model.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-5, 5)
    yy = a * xx - (model.intercept_[0]) / w[1]
    #Build boundaries and hyperplane
    margin = 1 / np.sqrt(np.sum(model.coef_ ** 2))
    yy_down = yy - np.sqrt(1 + a ** 2) * margin
    yy_up = yy + np.sqrt(1 + a ** 2) * margin
    #Plot Results
    plt.figure(figsize=(15, 8))
    plt.scatter(X[y==0, 0], X[y==0, 1], marker='o', c='red', label='Classe 1', s=100);
    plt.scatter(X[y==1, 0], X[y==1, 1], marker='o', c='blue', label='Classe -1', s=100);
    plt.plot(xx, yy, 'k-', label='Hiperplano de separação')
    plt.plot(xx, yy_down, 'b--', label='Limite de separação (-1)')
    plt.plot(xx, yy_up, 'r--', label='Limite de separação (1)')
    #Mark Support Vectors
    c = plt.Circle((model.support_vectors_[0, 0], model.support_vectors_[0, 1]), radius=0.15, color='k', fill=False, linewidth=3.0)
    plt.gca().add_patch(c)
    c = plt.Circle((model.support_vectors_[1, 0], model.support_vectors_[1, 1]), radius=0.15, color='k', fill=False, linewidth=3.0)
    plt.gca().add_patch(c)
    c = plt.Circle((model.support_vectors_[2, 0], model.support_vectors_[2, 1]), radius=0.15, color='k', fill=False, linewidth=3.0)
    plt.gca().add_patch(c)
    #Show Results
    plt.axis("equal")
    plt.legend(prop={'size': 18, 'weight':'bold'})
    plt.savefig('simpleSVM.png')
    plt.show()

def nonLinearData():
    #Create a random dataset and labels
    X = []
    y = []
    for _ in range(100):
        theta = np.random.uniform(0,2*np.pi)
        r = np.random.uniform(0.01,0.99)
        x1 = r * np.cos(theta)
        y1 = r * np.sin(theta)
        X.append([x1, y1])
        y.append(1)

        theta = np.random.uniform(0,2*np.pi)
        r = np.random.uniform(1.01,1.99)
        x1 = r * np.cos(theta)
        y1 = r * np.sin(theta)
        X.append([x1, y1])
        y.append(-1)
    X = np.array(X)
    y = np.array(y)
    #Plot dataset
    plt.figure(figsize=(15, 8))
    plt.scatter(X[y==-1, 0], X[y==-1, 1], marker='o', c='red', label='Classe 1', s=100);
    plt.scatter(X[y==1, 0], X[y==1, 1], marker='o', c='blue', label='Classe -1', s=100);
    #Plot class bondaries
    c = plt.Circle((0, 0), 1, color='blue', fill=False)
    plt.gca().add_patch(c)
    c = plt.Circle((0, 0), 2, color='red', fill=False)
    plt.gca().add_patch(c)
    #Plot Results
    plt.axis("equal")
    plt.legend(prop={'size': 18, 'weight':'bold'})
    plt.savefig('nonLinearData.png')
    plt.show()

def kernelPlot():
    #Create a random dataset and labels
    X = []
    y = []
    for _ in range(100):
        theta = np.random.uniform(0,2*np.pi)
        r = np.random.uniform(0.01,0.99)
        x1 = r * np.cos(theta)
        y1 = r * np.sin(theta)
        X.append([x1, y1])
        y.append(1)

        theta = np.random.uniform(0,2*np.pi)
        r = np.random.uniform(1.01,1.99)
        x1 = r * np.cos(theta)
        y1 = r * np.sin(theta)
        X.append([x1, y1])
        y.append(-1)
    X = np.array(X)
    y = np.array(y)
    #Applie Kernel function
    z = []
    for i in range(X.shape[0]):
        z.append(X[i,0]**2 + X[i,1]**2)
    z = np.array(z)
    #Plot dataset
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[y==-1,0], X[y==-1,1], z[y==-1], marker='o',label='Classe 1', s=100)
    ax.scatter(X[y==1,0], X[y==1,1], z[y==1], marker='o',label='Classe -1', s=100)
    #Plot hyperplane
    yy, xx = np.meshgrid(range(5), range(5))
    xx -= 2
    yy -= 2
    zz = np.ones(xx.shape)
    ax.plot_surface(xx, yy, zz, alpha=0.5, color='green')
    #Plot Results
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(elev=-170, azim=-45)
    plt.legend(prop={'size': 18, 'weight':'bold'})
    plt.savefig('posKernel.png')
    plt.show()

if __name__ == '__main__':
    #Fix Random Number Generation
    np.random.seed(12874)
    #exampleSVM()
    #nonLinearData()
    #kernelPlot()
    print(f"\nEnd Script!\n{'#'*50}")