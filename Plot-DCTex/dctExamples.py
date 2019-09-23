import numpy as np
import cv2
from matplotlib import pyplot as plt

def plotDCTex(grayImg):
    dctFeats = cv2.dct(np.array(grayImg, dtype=np.float32))
    #Thank's to https://stackoverflow.com/questions/50445847/how-to-zigzag-order-and-concatenate-the-value-every-line-using-python
    zigzagPattern = np.concatenate([np.diagonal(dctFeats[::-1,:], i)[::(2*(i % 2)-1)] for i in range(1-dctFeats.shape[0], dctFeats.shape[0])])
    cropZigzagPattern = zigzagPattern[0:1024]
    plt.figure(figsize=(15, 5))

    #************#

    plt.subplot(131)
    plt.imshow(src)
    plt.title('Imagem do Linfócito', fontname='Times New Roman', fontsize=18)
    
    #************#

    plt.subplot(132)
    spectrumPower = np.log10(abs(dctFeats))
    plt.imshow(spectrumPower, cmap='gray')
    plt.title('Transformada do Cosseno', fontname='Times New Roman', fontsize=18)

    #************#

    plt.subplot(133)
    zigzagEx = cv2.cvtColor(cv2.imread('zigzagScan.png'), cv2.COLOR_BGR2GRAY)
    plt.imshow(zigzagEx, cmap='gray')
    plt.title('ZigZag Scan', fontname='Times New Roman', fontsize=18)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('exDCTFeatsExtraction_0.png')
    plt.show()

    #************#

    plt.figure(figsize=(15, 5))
    
    with plt.style.context(('ggplot')):
        plt.subplot(121)
        plt.plot(zigzagPattern)
        plt.title(f'Coeficientes ordenados', fontname='Times New Roman', fontsize=18)

    #************#
    with plt.style.context(('ggplot')):
        plt.subplot(122)
        plt.plot(cropZigzagPattern)
        plt.title(f'Primeiros {len(cropZigzagPattern)} coeficientes', fontname='Times New Roman', fontsize=18)

    plt.tight_layout()
    plt.savefig('exDCTFeatsExtraction_1.png')
    plt.show()

    img0 = cv2.imread('exDCTFeatsExtraction_0.png')
    img1 = cv2.imread('exDCTFeatsExtraction_1.png')
    img = cv2.vconcat([img0, img1])
    cv2.imwrite('exDCTFeatsExtraction.png', img)
    
    

if __name__ == '__main__':
    #Fix Random Number Generation
    np.random.seed(12874)
    src = cv2.cvtColor(cv2.imread('UID_3_4_2_all.bmp'), cv2.COLOR_BGR2RGB)
    plotDCTex(cv2.cvtColor(src, cv2.COLOR_RGB2GRAY))
    print(f"\nEnd Script!\n{'#'*50}")
