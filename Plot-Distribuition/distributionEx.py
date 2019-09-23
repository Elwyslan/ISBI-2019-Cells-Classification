import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('ggplot')

if __name__ == '__main__':
    #Fix Random Number Generation
    np.random.seed(12874)
    data = 5*np.random.randn(1000) + 50
    plt.figure(figsize=(15, 8))
    plt.hist(data, bins=100)
    plt.axvline(x=np.percentile(data, 10), color='green', label='10º Percentil')
    plt.axvline(x=np.percentile(data, 25), color='yellow', label='25º Percentil')
    plt.axvline(x=np.median(data), color='blue', label='Mediana')
    plt.axvline(x=np.percentile(data, 75), color='indigo', label='75º Percentil')
    plt.axvline(x=np.percentile(data, 90), color='darkmagenta', label='90º Percentil')
    plt.axis("equal")
    plt.legend(prop={'size': 18, 'weight':'bold'})
    plt.savefig('exPercentis.png')
    plt.show()
    print(f"\nEnd Script!\n{'#'*50}")

"""
    img = cv2.imread(str(Path('../../train/fold_1/all/UID_2_8_1_all.bmp')))
    grayScaleImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    _,mask = cv2.threshold(grayScaleImage, 1, 1, cv2.THRESH_BINARY)

    for i in range(450):
        for j in range(450):
            if mask[i,j]==0:
                img[i,j]=255

    #plt.style.use('classic')
    fig, axarr = plt.subplots(1,2)
    fig.set_size_inches(10,10)
    axarr[0].set_xticks([])
    axarr[0].set_yticks([])
    axarr[0].imshow(img)
    axarr[1].set_xticks([])
    axarr[1].set_yticks([])
    axarr[1].imshow(mask, cmap='gray_r')
    #plt.savefig(Path('linfocito_e_mascara.png'))
    #plt.show()
    plt.clf()

    channelBLue = img[:,:,2]
    X = []
    for i in range(450):
        for j in range(450):
            if mask[i,j]==1:
                X.append(channelBLue[i,j])
    
    X = np.array(X)
    plt.hist(X, bins=[i for i in range(60,128,1)], rwidth=0.8)
    plt.axvline(np.median(X), color='red', linestyle='-', linewidth=2, label='Mediana')
    plt.axvline(np.percentile(X,10), color='darkgreen', linestyle='-', linewidth=2, label='10º percentil')
    plt.axvline(np.percentile(X,90), color='purple', linestyle='-', linewidth=2, label='90º percentil')
    plt.axvline(np.percentile(X,25), color='orange', linestyle='-', linewidth=2, label='25º percentil')
    plt.axvline(np.percentile(X,75), color='crimson', linestyle='-', linewidth=2, label='75º percentil')
    plt.xticks([])
    plt.yticks([])
    plt.legend(prop={'size': 18, 'weight':'bold'})
    plt.show()

    print(channelBLue.shape)
"""