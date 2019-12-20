################################################################################
# COMP3317 Computer Vision
# Assignment 4 - Camera calibration
################################################################################
import sys, argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d
from scipy.linalg import svd
from numpy.linalg import lstsq, qr, inv

################################################################################
#  estimate planar projective transformations for the 2 calibration planes
################################################################################
def calibrate2D(ref3D, ref2D):
    #  input:
    #    ref3D - a 8 x 3 numpy ndarray holding the 3D coodinates of the extreme
    #            corners on the 2 calibration planes
    #    ref2D - a 8 x 2 numpy ndarray holding the 2D coordinates of the projections
    #            of the corners in ref3D
    # return:
    #    Hxz - a 3 x 3 numpy ndarray holding the planar projective transformation
    #          for the X-Z plane
    #    Hyz - a 3 x 3 numpy ndarray holding the planar projective transformation
    #          for the Y-Z plane

    # TODO : form the matrix equation Ap = b for the X-Z plane


    xdiv = ref2D.shape[0]
    xs = xdiv // 2
    Axz = np.zeros((xdiv, 8))
    b = np.ones((xdiv, 1))

    for x in range(xs):
        X = ref3D[x][0]
        Z = ref3D[x][2]
        u = ref2D[x][0]
        v = ref2D[x][1]
        Axz[2*x] = np.array([X, Z, 1, 0, 0, 0, -u*X, -u*Z])
        Axz[2*x+1] = np.array([0, 0, 0, X, Z, 1, -v*X, -v*Z])
        b[2*x] = u
        b[2*x+1] = v


    # TODO : solve for the planar projective transformation using linear least squares

    m = np.linalg.lstsq(Axz, b, rcond = None)[0]
    m = np.append(m,1)
    Hxz=np.reshape(m, (3,3))

    # TODO : form the matrix equation Ap = b for the Y-Z plane
    Ayz = np.ones((xdiv, 8))

    for x in range(xs):
        Y = ref3D[x+4][1]
        Z = ref3D[x+4][2]
        u = ref2D[x+4][0]
        v = ref2D[x+4][1]
        Ayz[2*x] = np.array([Y, Z, 1, 0, 0, 0, -u*Y, -u*Z])
        Ayz[2*x+1] = np.array([0, 0, 0, Y, Z, 1, -v*Y, -v*Z])
        b[2*x] = u
        b[2*x+1] = v

    # TODO : solve for the planar projective transformation using linear least squares
    n = np.linalg.lstsq(Ayz, b, rcond = None)[0]
    n = np.append(n,1)
    Hyz = np.reshape(n, (3,3))

    return Hxz, Hyz

################################################################################
#  generate correspondences for all the corners on the 2 calibration planes
################################################################################
def gen_correspondences(Hxz, Hyz, corners) :
    # input:
    #    Hxz - a 3 x 3 numpy ndarray holding the planar projective transformation
    #          for the X-Z plane
    #    Hyz - a 3 x 3 numpy ndarray holding the planar projective transformation
    #          for the Y-Z plane
    #    corners - a n0 x 3 numpy ndarray holding the coordinates and strengths
    #              of the detected corners (n0 being the number of corners)
    # return:
    #    ref3D - a 160 x 3 numpy ndarray holding the 3D coodinates of all the corners
    #            on the 2 calibration planes
    #    ref2D - a 160 x 2 numpy ndarray holding the 2D coordinates of the projections
    #            of all the corners in ref3D

    # TODO : define 3D coordinates of all the corners on the 2 calibration planes
    newCoordXZ = np.zeros((corners.shape[0], 3))
    newCoordYZ = np.zeros((corners.shape[0], 3))

    for k in range(corners.shape[0]):
        su = corners[k][0] * corners[k][2]
        sv = corners[k][1] * corners[k][2]
        s = corners[k][2]
        coord = np.array([su, sv, s]).T
        coord3Dxz = inv(Hxz).dot(coord)
        coord3Dyz = inv(Hyz).dot(coord)


        coord3Dxz[0] = coord3Dxz[0]/coord3Dxz[2]
        coord3Dxz[1] = coord3Dxz[1]/coord3Dxz[2]
        coord3Dxz[2] = 1
        newCoordXZ[k] = coord3Dxz.T

        coord3Dyz[0]= coord3Dyz[0]/coord3Dyz[2]
        coord3Dyz[1]= coord3Dyz[1]/coord3Dyz[2]
        coord3Dyz[2]= 1
        newCoordYZ[k] = coord3Dyz.T


    # TODO : project corners on the calibration plane 1 onto the image
    cornersXZ2d = []
    cornersXZ3d = []
    for k in range(newCoordXZ.shape[0]):
        if(newCoordXZ[k][0] > 0 and newCoordXZ[k][1] > 0 and newCoordXZ[k][0] <= 10 and newCoordXZ[k][1] <= 8):
            transfo = Hxz.dot(newCoordXZ[k])
            transfo = transfo/transfo[2]
            cornersXZ2d.append(transfo[0:2])
            cornersXZ3d.append(newCoordXZ[k][0:2])

    cornersXZ2d = np.array(cornersXZ2d)
    cornersXZ3d = np.array(cornersXZ3d)
    vect1 = np.zeros((cornersXZ3d.shape[0],1))
    cornersXZ3d = np.hstack((cornersXZ3d[:,:1], vect1, cornersXZ3d[:,1:]))

    # TODO : project corners on the calibration plane 2 onto the image (YZ-plane)
    cornersYZ2d = []
    cornersYZ3d = []
    for k in range(newCoordYZ.shape[0]):
        if ((newCoordYZ[k][0] >  0)  and (newCoordYZ[k][1] > 0) and (newCoordYZ[k][0] <= 10 and (newCoordYZ[k][1] <= 8))):
            transfo = Hyz.dot(newCoordYZ[k])
            transfo = transfo/transfo[2]
            cornersYZ2d.append(transfo[0:2])
            cornersYZ3d.append(newCoordYZ[k][0:2])

    cornersYZ2d = np.array(cornersYZ2d)
    cornersYZ3d = np.array(cornersYZ3d)
    vect1 = np.zeros((cornersYZ3d.shape[0],1))
    cornersYZ3d = np.hstack((vect1,cornersYZ3d))

    # TODO : locate the nearest detected corners

    ref3D = np.concatenate((cornersXZ3d, cornersYZ3d), axis=0)
    ref2D = np.concatenate((cornersXZ2d, cornersYZ2d), axis=0)

    return ref3D, ref2D

################################################################################
#  estimate the camera projection matrix
################################################################################
def calibrate3D(ref3D, ref2D) :
    # input:
    #    ref3D - a 160 x 3 numpy ndarray holding the 3D coodinates of all the corners
    #            on the 2 calibration planes
    #    ref2D - a 160 x 2 numpy ndarray holding the 2D coordinates of the projections
    #            of all the corners in ref3D
    # output:
    #    P - a 3 x 4 numpy ndarray holding the camera projection matrix

    # TODO : form the matrix equation Ap = b for the camera

    xdiv = ref2D.shape[0]
    A = np.zeros((2*xdiv, 11))
    b = np.ones((2*xdiv, 1))

    for x in range(xdiv):
        X = ref3D[x][0]
        Y = ref3D[x][1]
        Z = ref3D[x][2]
        u = ref2D[x][0]
        v = ref2D[x][1]
        A[2*x] = np.array([X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z])
        A[2*x+1] = np.array([0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z])
        b[2*x] = u
        b[2*x+1] = v

    # TODO : solve for the projection matrix using linear least squares
    n = np.linalg.lstsq(A, b, rcond = None)[0]
    n = np.append(n,1)
    P = np.reshape(n, (3,4))

    return P

################################################################################
#  decompose the camera calibration matrix in K[R T]
################################################################################
def decompose_P(P) :
    # input:
    #    P - a 3 x 4 numpy ndarray holding the camera projection matrix
    # output:
    #    K - a 3 x 3 numpy ndarry holding the K matrix
    #    RT - a 3 x 4 numpy ndarray holding the rigid body transformation

    # TODO: extract the 3 x 3 submatrix from the first 3 columns of P
    P3 = P[:,3]
    P = np.delete(P,3,axis = 1)

    # TODO : perform QR decomposition on the inverse of [P0 P1 P2]
    p = np.linalg.inv(P)
    q, r = np.linalg.qr(p)

    # TODO : obtain K as the inverse of R
    K = np.linalg.inv(r)

    # TODO : obtain R as the tranpose of Q
    R = q.T

    # TODO : normalize K
    alpha = K[2][2]
    K = K/alpha

    k00 = K[0][0]
    if k00 < 0 :
        K[:,0] = -K[:,0]
        R[0,:] = -R[0,:]

    k11 = K[1][1]
    if k11 < 0:
        K[:,1] = -K[:,1]
        R[1,:] = -R[1,:]


    # TODO : obtain T from P3
    T = (1/alpha) * np.linalg.inv(K).dot(P3)
    T = np.reshape(T, (3,1))
    RT = np.hstack((R,T))

    return K, RT

################################################################################
#  check the planar projective transformations for the 2 calibration planes
################################################################################
def check_H(img_color, Hxz, Hyz) :
    # input:
    #    img_color - a h x w x 3 numpy ndarray (dtype = np.unit8) holding
    #                the color image (h, w being the height and width of the image)
    #    Hxz - a 3 x 3 numpy ndarray holding the planar projective transformation
    #          for the X-Z plane
    #    Hyz - a 3 x 3 numpy ndarray holding the planar projective transformation
    #          for the Y-Z plane

    # plot the image
    plt.ion()
    fig = plt.figure('Camera calibration')
    plt.imshow(img_color)

    # define 3D coordinates of all the corners on the 2 calibration planes
    X_ = np.arange(10) + 0.5 # Y == X
    Z_ = np.arange(8) + 0.5
    X_ = np.tile(X_, 8)
    Z_ = np.repeat(Z_, 10)
    X = np.vstack((X_, Z_, np.ones(80)))

    # project corners on the calibration plane 1 onto the image
    w = Hxz @ X
    u = w[0, :] / w[2, :]
    v = w[1, :] / w[2, :]
    plt.plot(u, v, 'r.', markersize = 3)

    # project corners on the calibration plane 2 onto the image
    w = Hyz @ X
    u = w[0, :] / w[2, :]
    v = w[1, :] / w[2, :]
    plt.plot(u, v, 'r.', markersize = 3)

    plt.show()
    plt.ginput(n = 1, timeout = - 1)
    plt.close(fig)

################################################################################
#  check the 2D correspondences for the 2 calibration planes
################################################################################
def check_correspondences(img_color, ref2D) :
    # input:
    #    img_color - a h x w x 3 numpy ndarray (dtype = np.unit8) holding
    #                the color image (h, w being the height and width of the image)
    #    ref2D - a 160 x 2 numpy ndarray holding the 2D coordinates of the projections
    #            of all the corners on the 2 calibration planes

    # plot the image and the correspondences
    plt.ion()
    fig = plt.figure('Camera calibration')
    plt.imshow(img_color)
    plt.plot(ref2D[:, 0], ref2D[:, 1], 'bx', markersize = 5)
    plt.show()
    plt.ginput(n = 1, timeout = - 1)
    plt.close(fig)

################################################################################
#  check the estimated camera projection matrix
################################################################################
def check_P(img_color, ref3D, P) :
    # input:
    #    img_color - a h x w x 3 numpy ndarray (dtype = np.unit8) holding
    #                the color image (h, w being the height and width of the image)
    #    ref3D - a 160 x 3 numpy ndarray holding the 3D coodinates of all the corners
    #            on the 2 calibration planes
    #    P - a 3 x 4 numpy ndarray holding the camera projection matrix

    # plot the image
    plt.ion()
    fig = plt.figure('Camera calibration')
    plt.imshow(img_color)

    # project the reference 3D points onto the image
    w = P @ np.append(ref3D, np.ones([len(ref3D), 1]), axis = 1).T
    u = w[0, :] / w[2, :]
    v = w[1, :] / w[2, :]
    plt.plot(u, v, 'r.', markersize = 3)
    plt.show()
    plt.ginput(n = 1, timeout = - 1)
    plt.close(fig)

################################################################################
#  pick seed corners for camera calibration
################################################################################
def pick_corners(img_color, corners) :
    # input:
    #    img_color - a h x w x 3 numpy ndarray (dtype = np.unit8) holding
    #                the color image (h, w being the height and width of the image)
    #    corners - a n x 3 numpy ndarray holding the coordinates and strengths
    #              of the detected corners (n being the number of corners)
    # return:
    #    ref3D - a 8 x 3 numpy ndarray holding the 3D coodinates of the extreme
    #            corners on the 2 calibration planes
    #    ref2D - a 8 x 2 numpy ndarray holding the 2D coordinates of the projections
    #            of the corners in ref3D

    # plot the image and corners
    plt.ion()
    fig = plt.figure('Camera calibration')
    plt.imshow(img_color)
    plt.plot(corners[:,0], corners[:,1],'r+', markersize = 5)
    plt.show()

    # define 3D coordinates of the extreme corners on the 2 calibration planes
    ref3D = np.array([(9.5, 0, 7.5), (0.5, 0, 7.5), (9.5, 0, 0.5), (0.5, 0, 0.5),
                      (0, 0.5, 7.5), (0, 9.5, 7.5), (0, 0.5, 0.5), (0, 9.5, 0.5)],
                      dtype = np.float64)
    ref2D = np.zeros([8, 2], dtype = np.float64)
    for i in range(8) :
        selected = False
        while not selected :
            # ask user to pick the corner on the image
            print('please click on the image point for ({}, {}, {})...'.format(
                  ref3D[i, 0], ref3D[i, 1], ref3D[i, 2]))
            plt.figure(fig.number)
            pt = plt.ginput(n = 1, timeout = - 1)
            # locate the nearest detected corner
            pt = find_nearest_corner(np.array(pt), corners)
            if pt[0, 0] > 0 :
                plt.figure(fig.number)
                plt.plot(pt[:, 0], pt[:, 1], 'bx', markersize = 5)
                ref2D[i, :] = pt[0]
                selected = True
            else :
                print('cannot locate detected corner in the vicinity...')
    plt.close(fig)

    return ref3D, ref2D

################################################################################
#  find nearest corner
################################################################################
def find_nearest_corner(pts, corners) :
    # input:
    #    pts - a n0 x 2 numpy ndarray holding the coordinates of the points
    #          (n0 being the number of points)
    #    corners - a n1 x 3 numpy ndarray holding the coordinates and strengths
    #              of the detected corners (n1 being the number of corners)
    # return:
    #    selected - a n0 x 2 numpy ndarray holding the coordinates of the nearest_corner
    #               corner

    x = corners[:, 0]
    y = corners[:, 1]
    x_ = pts[:, 0]
    y_ = pts[:, 1]

    # compute distances between the corners and the pts
    dist = np.sqrt(np.square(x.reshape(-1,1).repeat(len(x_), axis = 1) - x_)
                 + np.square(y.reshape(-1,1).repeat(len(y_), axis = 1) - y_))
    # find the index of the corner with the min distance for each pt
    min_idx = np.argmin(dist, axis = 0)
    # find the min distance for each pt
    min_dist = dist[min_idx, np.arange(len(x_))]
    # extract the nearest corner for each pt
    selected = corners[min_idx, 0:2]
    # identify corners with a min distance > 10 and replace them with [-1, -1]
    idx = np.where(min_dist > 10)
    selected[idx, :] = [-1 , -1]
    return selected

################################################################################
#  save K[R T] to a file
################################################################################
def save_KRT(outputfile, K, RT) :
    # input:
    #    outputfile - path of the output file
    #    K - a 3 x 3 numpy ndarry holding the K matrix
    #    RT - a 3 x 4 numpy ndarray holding the rigid body transformation

    try :
        file = open(outputfile, 'w')
        for i in range(3) :
            file.write('{:.6e} {:.6e} {:.6e}\n'.format(K[i,0], K[i, 1], K[i, 2]))
        for i in range(3) :
            file.write('{:.6e} {:.6e} {:.6e} {:.6e}\n'.format(RT[i, 0], RT[i, 1],
                       RT[i, 2], RT[i, 3]))
        file.close()
    except :
        print('Error occurs in writting output to \'{}\'.'.format(outputfile))
        sys.exit(1)

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
#  load corners from a file
################################################################################
def load_corners(inputfile) :
    # input:
    #    inputfile - path of the file containing corner detection output
    # return:
    #    corners - a n x 3 numpy ndarray holding the coordinates and strengths
    #              of the detected corners (n being the number of corners)

    try :
        file = open(inputfile, 'r')
        line = file.readline()
        nc = int(line.strip())
        # print('loading {} corners'.format(nc))
        corners = np.zeros([nc, 3], dtype = np.float64)
        for i in range(nc) :
            line = file.readline()
            x, y, r = line.split()
            corners[i] = [np.float64(x), np.float64(y), np.float64(r)]
        file.close()
        return corners
    except :
        print('Error occurs in loading corners from \'{}\'.'.format(inputfile))
        sys.exit(1)

################################################################################
#  main
################################################################################
def main() :
    parser = argparse.ArgumentParser(description = 'COMP3317 Assignment 4')
    parser.add_argument('-i', '--image', type = str, default = 'grid1.jpg',
                        help = 'filename of input image')
    parser.add_argument('-c', '--corners', type = str, default = 'grid1.crn',
                        help = 'filename of corner detection output')
    parser.add_argument('-o', '--output', type = str, default = 'grid1.cam',
                        help = 'filename for outputting camera calibration result')
    args = parser.parse_args()

    print('-------------------------------------------')
    print('COMP3317 Assignment 4 - Camera calibration')
    print('input image : {}'.format(args.image))
    print('corner list : {}'.format(args.corners))
    print('output file : {}'.format(args.output))
    print('-------------------------------------------')

    # load the image
    img_color = load_image(args.image)
    print('\'{}\' loaded...'.format(args.image))

    # load the corner detection result
    corners = load_corners(args.corners)
    print('{} corners loaded from \'{}\'...'.format(len(corners), args.corners))

    # pick the seed corners for camera calibration
    print('pick seed corners for camera calibration...')
    ref3D, ref2D = pick_corners(img_color, corners)

    # estimate planar projective transformations for the 2 calibration planes
    print('estimate planar projective transformations for the 2 calibration planes...')
    H1, H2 = calibrate2D(ref3D, ref2D)
    check_H(img_color, H1, H2)

    # generate correspondences for all the corners on the 2 calibration planes
    print('generate correspondences for all the corners on the 2 calibration planes...')
    ref3D, ref2D = gen_correspondences(H1, H2, corners)
    check_correspondences(img_color, ref2D)

    # estimate the camera projection matrix
    print('estimate the camera projection matrix...')
    P = calibrate3D(ref3D, ref2D)
    print('P = ')
    print(P)
    check_P(img_color, ref3D, P)

    # decompose the camera projection matrix into K[R T]
    print('decompose the camera projection matrix...')
    K, RT = decompose_P(P)
    print('K =')
    print(K)
    print('[R T] =')
    print(RT)
    check_P(img_color, ref3D, K @ RT)

    # save K[R T] to a file
    if args.output :
        save_KRT(args.output, K, RT)
        print('K[R T] saved to \'{}\'...'.format(args.output))

if __name__ == '__main__':
    main()
