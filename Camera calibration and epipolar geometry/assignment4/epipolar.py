################################################################################
# COMP3317 Computer Vision
# Assignment 4 - Epipolar constraint
################################################################################
import sys, argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d
from numpy.linalg import lstsq, qr, inv, pinv

################################################################################
#  compose the essential matrix from proj matrices
################################################################################
def compose_E(RT1, RT2) :
    # input:
    #    RT1 - a 3 x 4 numpy ndarray holding the rigid body transformation
    #          of camera 1
    #    RT2 - a 3 x 4 numpy ndarray holding the rigid body transformation
    #          of camera 2
    # output:
    #    E - a 3 x 3 numpy ndarray holding the essential matrix

    # TODO : compute the relative rotation R
    R1 = RT1[:, 0:3]
    R2 = RT2[:, 0:3]
    R = np.linalg.inv(R1).dot(R2)

    # TODO: compute the relative translation T
    T1 = RT1[:, 3]
    T2 = RT2[:, 3]

    T = np.linalg.inv(R1).dot(T2 - T1)

    # TODO : compose E from R and T
    Tx = np.array([[0, -T[2], T[1]], [T[2], 0, -T[0]], [-T[1], T[0], 0]])
    E = Tx.dot(R)

    return E

################################################################################
#  check the essential matrix
################################################################################
def check_E(img1, file1, img2, file2, E, K1, K2) :
    # input:
    #    img1 - a h x w x 3 numpy ndarray (dtype = np.unit8) holding the color
    #           image 1 (h, w being the height and width of the image)
    #    file1 - the filename of the color image 1
    #    img2 - a h x w x 3 numpy ndarray (dtype = np.unit8) holding the color
    #           image 2 (h, w being the height and width of the image)
    #    filen2 - the filename of the color image 2
    #    E - a 3 x 3 numpy ndarray holding the essential matrix
    #    K1 - a 3 x 3 numpy ndarray holding the K matrix of camera 1
    #    K2 - a 3 x 3 numpy ndarray holding the K matrix of camera 2

    # form the fundamental matrix
    F = inv(K2.T) @ E @ inv(K1)

    plt.ion()
    # plot image 1
    fig1 = plt.figure('Epipolar geometry - {}'.format(file1))
    plt.imshow(img1)
    plt.autoscale(enable = False, axis = 'both')
    # plot image 2
    fig2 = plt.figure('Epipolar geometry - {}'.format(file2))
    plt.imshow(img2)
    plt.autoscale(enable = False, axis = 'both')
    plt.show()

    # ask user to pick points on image 1
    print('please select points on {} using left-click, end with right-click...'.format(file1))
    plt.figure(fig1.number)
    pt = plt.ginput(n = 1, mouse_pop = 2, mouse_stop = 3, timeout = -1)
    while pt :
        pt = np.array(pt)
        plt.figure(fig1.number)
        plt.plot(pt[0, 0], pt[0, 1], 'rx')
        # form the epipolar line
        (a, b, c) = F @ np.array((pt[0, 0], pt[0, 1], 1))
        (h, w, d) = img2.shape
        x = np.array([0, w - 1])
        y = np.array([(a * x[0] + c) / (-b), (a * x[1] + c) / (-b)])
        if y[0] < 0 or y[0] > h - 1 or y[1] < 0 or y[1] > h - 1 :
            y = np.array([0, h - 1])
            x = np.array([(b * y[0] + c) / (-a), (b * y[1] + c) / (-a)])
        plt.figure(fig2.number)
        plt.plot(x, y, 'r-')
        plt.figure(fig1.number)
        pt = plt.ginput(n = 1, mouse_pop = 2, mouse_stop = 3, timeout = -1)

    # ask user to pick points on image 1
    print('please select points on {} using left-click, end with right-click...'.format(file2))
    plt.figure(fig2.number)
    pt = plt.ginput(n = 1, mouse_pop = 2, mouse_stop = 3, timeout = -1)
    while pt :
        pt = np.array(pt)
        plt.figure(fig2.number)
        plt.plot(pt[0, 0], pt[0, 1], 'bx')
        # form the epipolar line
        (a, b, c) = F.T @ np.array((pt[0, 0], pt[0, 1], 1))
        (h, w, d) = img1.shape
        x = np.array([0, w - 1])
        y = np.array([(a * x[0] + c) / (-b), (a * x[1] + c) / (-b)])
        if y[0] < 0 or y[0] > h - 1 or y[1] < 0 or y[1] > h - 1 :
            y = np.array([0, h - 1])
            x = np.array([(b * y[0] + c) / (-a), (b * y[1] + c) / (-a)])
        plt.figure(fig1.number)
        plt.plot(x, y, 'b-')
        plt.figure(fig2.number)
        pt = plt.ginput(n = 1, mouse_pop = 2, mouse_stop = 3, timeout = -1)
    plt.close()

################################################################################
#  save E to a file
################################################################################
def save_E(outputfile, E) :
    # input:
    #    outputfile - path of the output file
    #    E - a 3 x 3 numpy ndarry holding the essential matrix

    try :
        file = open(outputfile, 'w')
        for i in range(3) :
            file.write('{:.6e} {:.6e} {:.6e}\n'.format(E[i,0], E[i, 1], E[i, 2]))
        file.close()
    except :
        print('Error occurs in writting output to \'{}\'.'.format(outputfile))
        sys.exit(1)

################################################################################
#  load E from a file
################################################################################
def load_E(inputfile) :
    # input:
    #    inputfile - path of the file containing E
    # return:
    #    E - a 3 x 3 numpy ndarry holding the essential matrix

    try :
        file = open(inputfile, 'r')
        K = np.zeros([3, 3], dtype = np.float64)
        for i in range(3) :
            line = file.readline()
            e0, e1, e2 = line.split()
            E[i] = [np.float64(e0), np.float64(e1), np.float64(e2)]
        file.close()
    except :
        print('Error occurs in loading E from \'{}\'.'.format(inputfile))
        sys.exit(1)

    return E

################################################################################
#  load K[R T] from a file
################################################################################
def load_KRT(inputfile) :
    # input:
    #    inputfile - path of the file containing K[R T]
    # return:
    #    K - a 3 x 3 numpy ndarry holding the K matrix
    #    RT - a 3 x 4 numpy ndarray holding the rigid body transformation

    try :
        file = open(inputfile, 'r')
        K = np.zeros([3, 3], dtype = np.float64)
        RT = np.zeros([3, 4], dtype = np.float64)
        for i in range(3) :
            line = file.readline()
            e0, e1, e2 = line.split()
            K[i] = [np.float64(e0), np.float64(e1), np.float64(e2)]
        for i in range(3) :
            line = file.readline()
            e0, e1, e2, e3 = line.split()
            RT[i] = [np.float64(e0), np.float64(e1), np.float64(e2), np.float64(e3)]
        file.close()
    except :
        print('Error occurs in loading K[R T] from \'{}\'.'.format(inputfile))
        sys.exit(1)

    return K, RT

################################################################################
#  load image from a file
################################################################################
def load_image(inputfile) :
    # input:
    #    inputfile - path of the image file
    # return:
    #    img_color - a h x w x 3 numpy ndarray (dtype = np.unit8) holding
    #                the color image (h, w being the height and width of the image)

    try :
        img_color = plt.imread(inputfile)
        return img_color
    except :
        print('Cannot open \'{}\'.'.format(inputfile))
        sys.exit(1)

################################################################################
#  main
################################################################################
def main() :
    parser = argparse.ArgumentParser(description = 'COMP3317 Assignment 4')
    parser.add_argument('-i1', '--image1', type = str, default = 'grid1.jpg',
                        help = 'filename of input image 1')
    parser.add_argument('-c1', '--cam1', type = str, default = 'grid1.cam',
                        help = 'filename of camera calibration output 1')
    parser.add_argument('-i2', '--image2', type = str, default = 'grid2.jpg',
                        help = 'filename of input image 2')
    parser.add_argument('-c2', '--cam2', type = str, default = 'grid2.cam',
                        help = 'filename of camera calibration output 2')
    parser.add_argument('-o', '--output', type = str,
                        help = 'filename for outputting essential matrix')
    args = parser.parse_args()

    print('-----------------------------------------')
    print('COMP3317 Assignment 4 - Eipolar geometry')
    print('input image 1 : {}'.format(args.image1))
    print('proj martix 1 : {}'.format(args.cam1))
    print('input image 2 : {}'.format(args.image2))
    print('proj matrix 2 : {}'.format(args.cam2))
    print('output file   : {}'.format(args.output))
    print('-----------------------------------------')

    # load image 1 and proj matrix 1
    img_color1 = load_image(args.image1)
    print('\'{}\' loaded...'.format(args.image1))
    K1, RT1 = load_KRT(args.cam1)
    print('\'{}\' loaded...'.format(args.cam1))

    # load image 2 and proj matrix 2
    img_color2 = load_image(args.image2)
    print('\'{}\' loaded...'.format(args.image2))
    K2, RT2 = load_KRT(args.cam2)
    print('\'{}\' loaded...'.format(args.cam2))

    # compose the essential matrix from the proj matrices
    E = compose_E(RT1, RT2)
    print('E = ')
    print(E)
    check_E(img_color1, args.image1, img_color2, args.image2, E, K1, K2)

    # save E to a file
    if args.output :
        save_E(args.output, E)
        print('E saved to \'{}\'...'.format(args.output))
        # check saved data
        # E = load_E(args.output)
        # print('E = ')
        # print(E)

if __name__ == '__main__':
    main()
