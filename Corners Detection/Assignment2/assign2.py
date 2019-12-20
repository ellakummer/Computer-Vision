################################################################################
# COMP3317 Computer Vision
# Assignment 2 - Conrner detection
################################################################################
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d

###############################################################################
#  perform RGB to grayscale conversion
################################################################################

def rgb2gray(img_color) :
    # input:
    #    img_color - a h x w x 3 numpy ndarray (dtype = np.unit8) holding
    #                the color image
    # return:
    #    img_gray - a h x w numpy ndarray (dtype = np.float64) holding
    #               the grayscale image
    #
    # TODO: using the Y channel of the YIQ model to perform the conversion
    # récupérer les dimensions
    sizeImage = img_color.shape
    x = sizeImage[0]
    y = sizeImage[1]
    # créer une matrice carrée qui sera celle de retour (full de 0)
    img_gray = np.zeros((x,y))
    # appliquer la transformation à chaque pixel de la matrice
    # modifier la valeur dans la matrice de retour
    for i in range(x):
        for j in range(y):
            # weight of pixels :
            r = img_color[i,j,0]
            g = img_color[i,j,1]
            b = img_color[i,j,2]
            img_gray[i,j] = 0.299*r + 0.587*g + 0.114*b

    print("dimensions: ", img_gray.shape)

    return img_gray

################################################################################
#  perform 1D smoothing using a 1D horizontal Gaussian filter
################################################################################
def smooth1D(img, sigma) :
    # input :
    #    img - a h x w numpy ndarray holding the image to be smoothed
    #    sigma - sigma value of the 1D Gaussian function
    # return:
    #    img_smoothed - a h x w numpy ndarry holding the 1D smoothing result

    # TODO: form a 1D horizontal Gaussian filter of an appropriate size
    n = 3*sigma  # : size (slide 31, lecture 3)
    x = np.arange(-1*n, n+1)

    # TODO: convolve the 1D filter with the image;
    #       apply partial filter for the image border

    filter = np.exp((x**2)/-2/(sigma**2))

    img_smoothed = convolve1d(img,filter, 1, np.float64,'constant',0,0)

    # TODO : normalization

    sizeImage2 = img_smoothed.shape
    x2 = sizeImage2[0]
    y2 = sizeImage2[1]
    norma = np.ones((x2,y2))
    weight = convolve1d(norma,filter, 1, np.float64,'constant',0,0)
    img_smoothed3 = np.divide(img_smoothed,weight)

    return img_smoothed3

################################################################################
#  perform 2D smoothing using 1D convolutions
################################################################################
def smooth2D(img, sigma) :
    # input:
    #    img - a h x w numpy ndarray holding the image to be smoothed
    #    sigma - sigma value of the Gaussian function
    # return:
    #    img_smoothed - a h x w numpy array holding the 2D smoothing result

    # TODO: smooth the image along the vertical direction
    smoothVertical = smooth1D(img,sigma)
    # TODO: smooth the image along the horizontal direction
    img_smoothed = smooth1D(smoothVertical.transpose(),sigma)
    img_smoothed = img_smoothed.transpose()

    return img_smoothed

################################################################################
#   perform Harris corner detection
################################################################################
def harris(img, sigma, threshold) :
# input:
    #    img - a h x w numpy ndarry holding the input image
    #    sigma - sigma value of the Gaussian function used in smoothing
    #    threshold - threshold value used for identifying corners
    # return:
    #    corners - a list of tuples (x, y, r) specifying the coordinates
    #              (up to sub-pixel accuracy) and cornerness value of each corner

    # TODO: compute Ix & Iy

    diff = np.array([0.5, 0, -0.5])
    Ix = np.ones((img.shape[0],img.shape[1]))
    Ix = convolve1d(img,diff, 1, np.float64,'constant',0,0)
    Iy = np.ones((img.shape[0], img.shape[1]))
    Iy = convolve1d(img.T, diff, 1, np.float64, 'constant', 0, 0)
    Iy = Iy.T


    # TODO: compute Ix2, Iy2 and IxIy

    Ix2= np.square(Ix)
    Iy2= np.square(Iy)
    Ixy= np.multiply(Ix,Iy)

    # TODO: smooth the squared derivatives
    sIx2 = smooth2D(Ix2,sigma)
    sIy2 = smooth2D(Iy2,sigma)
    sIxy = smooth2D(Ixy,sigma)

    # TODO: compute cornesness function R

    k = 0.04
    R = ((sIx2*sIy2)-(sIxy*sIxy)) - (k*((sIx2+sIy2)**2))

    corners = []

    # TODO: mark local maxima as corner candidates;
    #       perform quadratic approximation to local corners upto sub-pixel accuracy

    for m in range(1,R.shape[0]-1) :
        for n in range(1,R.shape[1]-1) :
            x = np.array([R[m-1,n-1], R[m-1,n], R[m-1,n+1], R[m,n-1], R[m,n], R[m,n+1], R[m+1,n-1], R[m+1,n], R[m+1,n+1]])

            if R[m,n] == np.amax(x):
                a = (R[m-1,n] + R[m+1,n] - (2*R[m,n]))/2
                b = (R[m,n-1] + R[m,n+1] - (2*R[m,n]))/2
                c = (R[m+1,n] - R[m-1,n])/2
                d = (R[m,n+1] - R[m,n-1])/2
                e = R[m,n]
                # TODO: perform thresholding and discard weak corners
                if R[m,n] > threshold:
                    m_subpixel = m-(c/(2*a))
                    n_subpixel = n-(d/(2*b))
                    corners.append((n_subpixel, m_subpixel, R[m,n]))


    #print(corners)
    #print(R)

    return sorted(corners, key = lambda corner : corner[2], reverse = True)


################################################################################
#   save corners to a file
################################################################################
def save(outputfile, corners) :
    try :
        file = open(outputfile, 'w')
        file.write('%d\n' % len(corners))
        for corner in corners :
            file.write('%.4f %.4f %.4f\n' % corner)
        file.close()
    except :
        print('Error occurs in writting output to \'%s\''  % outputfile)
        sys.exit(1)

################################################################################
#   load corners from a file
################################################################################
def load(inputfile) :
    try :
        file = open(inputfile, 'r')
        line = file.readline()
        nc = int(line.strip())
        print('loading %d corners' % nc)
        corners = list()
        for i in range(nc) :
            line = file.readline()
            (x, y, r) = line.split()
            corners.append((float(x), float(y), float(r)))
        file.close()
        return corners
    except :
        print('Error occurs in writting output to \'%s\''  % outputfile)
        sys.exit(1)

################################################################################
## main
################################################################################
def main() :
    parser = argparse.ArgumentParser(description = 'COMP3317 Assignment 2')
    parser.add_argument('-i', '--inputfile', type = str, default = 'grid1.jpg', help = 'filename of input image')
    parser.add_argument('-s', '--sigma', type = float, default = 1.0, help = 'sigma value for Gaussain filter')
    parser.add_argument('-t', '--threshold', type = float, default = 1e6, help = 'threshold value for corner detection')
    parser.add_argument('-o', '--outputfile', type = str, help = 'filename for outputting corner detection result')
    args = parser.parse_args()

    print('------------------------------')
    print('COMP3317 Assignment 2')
    print('input file : %s' % args.inputfile)
    print('sigma      : %.2f' % args.sigma)
    print('threshold  : %.2e' % args.threshold)
    print('output file: %s' % args.outputfile)
    print('------------------------------')

    # load the image
    try :
        #img_color = imageio.imread(args.inputfile)
        img_color = plt.imread(args.inputfile)
        print('%s loaded...' % args.inputfile)
    except :
        print('Cannot open \'%s\'.' % args.inputfile)
        sys.exit(1)
    # uncomment the following 2 lines to show the color image
    # plt.imshow(np.uint8(img_color))
    # plt.show()

    # perform RGB to gray conversion
    print('perform RGB to grayscale conversion...')
    img_gray = rgb2gray(img_color)
    # uncomment the following 2 lines to show the grayscale image
    # plt.imshow(np.float32(img_gray), cmap = 'gray')
    # plt.show()

    # perform corner detection
    print('perform Harris corner detection...')
    corners = harris(img_gray, args.sigma, args.threshold)

    # plot the corners
    print('%d corners detected...' % len(corners))
    x = [corner[0] for corner in corners]
    y = [corner[1] for corner in corners]
    fig = plt.figure()
    plt.imshow(np.float32(img_gray), cmap = 'gray')
    plt.plot(x, y,'r+',markersize = 5)
    plt.show()

    # save corners to a file
    if args.outputfile :
        save(args.outputfile, corners)
        print('corners saved to \'%s\'...' % args.outputfile)

if __name__ == '__main__':
    main()
