import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.spatial import distance
from scipy.signal import resample
from scipy.interpolate import spline

def plotContourSignature(grayScaleImage, sizeVector=256):
    _,thresh = cv2.threshold(grayScaleImage, 1, 1, cv2.THRESH_BINARY)
    _,invThresh = cv2.threshold(grayScaleImage, 1, 1, cv2.THRESH_BINARY_INV)
    _, contours, _ = cv2.findContours(thresh, 1, 2)
    contour = max(contours, key=cv2.contourArea)
    m = cv2.moments(thresh)
    centroid = (int(m['m10']/m['m00']), int(m['m01']/m['m00']))
    signature = []
    for points in contour:
        curvePoint = points[0][0], points[0][1]
        signature.append(distance.euclidean(centroid, curvePoint))
    if (len(signature)>sizeVector) or (len(signature)<sizeVector):
        signature = resample(signature, sizeVector)

    #signature = signature - np.mean(signature)
    fourierCoefs = list(map(abs, np.fft.fft(signature)))

    ##################################################################
    plt.figure(figsize=(15, 8))
    
    plt.subplot(221)
    plt.imshow(src)
    plt.title('Imagem do Linfócito', fontname='Times New Roman', fontsize=18)

    #************#

    contourPoints = cv2.drawContours(grayScaleImage, [contour], 0, 255, 5) * invThresh
    cv2.drawMarker(contourPoints,centroid, 255, markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
    for i in [20, 25, 30, 35, 40]:
        cv2.line(contourPoints, centroid, tuple(contour[i][0]), 255, 2)
    plt.subplot(222)
    plt.imshow(contourPoints,cmap='gray')
    plt.title('Contorno do linfócito', fontname='Times New Roman', fontsize=18)
    
    #************#

    with plt.style.context(('ggplot')):
        plt.subplot(223)
        plt.plot(signature)
        plt.title('Função Distância do Centróide', fontname='Times New Roman', fontsize=18)

    #************#

    with plt.style.context(('ggplot')):
        plt.subplot(224)
        logCoefs = np.log10(fourierCoefs[0:sizeVector//2])
        #plt.plot(logCoefs)
        plt.bar([i for i in range(len(logCoefs))], logCoefs, width=0.6)
        plt.title('Coeficientes de Fourier (log)', fontname='Times New Roman', fontsize=18)

    plt.tight_layout()
    plt.savefig('fourierShapeDescriptor.png')
    plt.show()

def plotContourFeatures(grayScaleImage, sizeVector=320):
    _,thresh = cv2.threshold(grayScaleImage, 1, 1, cv2.THRESH_BINARY)
    _,invThresh = cv2.threshold(grayScaleImage, 1, 1, cv2.THRESH_BINARY_INV)
    _, contours, _ = cv2.findContours(thresh, 1, 2)
    contour = max(contours, key=cv2.contourArea)
    m = cv2.moments(thresh)
    centroid = (int(m['m10']/m['m00']), int(m['m01']/m['m00']))
    signature = []
    for points in contour:
        curvePoint = points[0][0], points[0][1]
        signature.append(distance.euclidean(centroid, curvePoint))
    if (len(signature)>sizeVector) or (len(signature)<sizeVector):
        origiSignature = signature.copy()
        signature = resample(signature, sizeVector)
    
    fourierCoefs = list(map(abs, np.fft.fft(signature)))

    ##################################################################

    plt.figure(figsize=(15, 8))
    
    plt.subplot(121)
    plt.imshow(src)
    plt.title('Imagem do Linfócito', fontname='Times New Roman', fontsize=55)

    #************#

    contourPoints = cv2.drawContours(grayScaleImage, [contour], 0, 255, 5) * invThresh
    cv2.drawMarker(contourPoints,centroid, 255, markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
    #for i in [20, 25, 30, 35, 40]:
        #cv2.line(contourPoints, centroid, tuple(contour[i][0]), 255, 2)
    plt.subplot(122)
    plt.imshow(contourPoints,cmap='gray')
    plt.title('Contorno do Linfócito', fontname='Times New Roman', fontsize=55)

    plt.tight_layout()
    plt.savefig('exContourFeatsExtraction-a.png')
    plt.show()

    ################################################

    plt.figure(figsize=(15, 8))

    with plt.style.context(('ggplot')):
        plt.subplot(221)
        plt.plot(origiSignature)
        plt.title(f'Função Distância do Centróide ({len(origiSignature)} amostras)', fontname='Times New Roman', fontsize=18)

    #************#

    with plt.style.context(('ggplot')):
        plt.subplot(222)
        plt.plot(signature)
        plt.title(f'Função Distância do Centróide após o Resampling ({len(signature)} amostras)', fontname='Times New Roman', fontsize=18)

    #************#

    with plt.style.context(('ggplot')):
        plt.subplot(223)
        logCoefs = np.log10(fourierCoefs)
        #plt.plot(logCoefs)
        plt.bar([i for i in range(len(logCoefs))], logCoefs, width=0.5)
        plt.title('Coeficientes de Fourier', fontname='Times New Roman', fontsize=18)

    #************#

    with plt.style.context(('ggplot')):
        plt.subplot(224)
        logCoefs = np.log10(fourierCoefs[0:sizeVector//2])
        #plt.plot(logCoefs)
        plt.bar([i for i in range(len(logCoefs))], logCoefs, width=0.5)
        plt.title('Primeira metade dos coeficientes de Fourier', fontname='Times New Roman', fontsize=18)

    plt.tight_layout()
    plt.savefig('exContourFeatsExtraction-b.png')
    plt.show()


    

if __name__ == '__main__':
    #Fix Random Number Generation
    np.random.seed(12874)
    src = cv2.cvtColor(cv2.imread('UID_H10_47_2_hem.bmp'), cv2.COLOR_BGR2RGB)
    plotContourSignature(cv2.cvtColor(src, cv2.COLOR_RGB2GRAY))
    plotContourFeatures(cv2.cvtColor(src, cv2.COLOR_RGB2GRAY))
    print(f"\nEnd Script!\n{'#'*50}")
