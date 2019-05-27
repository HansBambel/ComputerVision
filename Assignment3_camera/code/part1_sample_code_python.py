from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy


def fit_fundamental_matrix(matches):
    print("Fitting of the fundamental Matrix takes place here!!!")
    equations = []
    for m in matches:
        x1 = m[0]
        y1 = m[1]
        x2 = m[2]
        y2 = m[3]
        equations.append(np.array([x1*x2, x1*y2, x1, x2*y1, y1*y2, y1, x2, y2, 1]))

    # Slide 97 --> Matrix A.shape = 309x9
    equations = np.array(equations)
    u, d, v = np.linalg.svd(equations)
    # We got UDV here

    # get entries for F from smallest singular value in d
    entries_F = v[np.argmin(d)]
    # do another SVD with respect to F
    u_f, d_f, v_f = np.linalg.svd(entries_F.reshape(3, 3))
    # set smallest value to 0 --> to enforce Rank 2
    d_f[np.argmin(d_f)] = 0
    d_f_prime = np.zeros((len(d_f), len(d_f)))
    np.fill_diagonal(d_f_prime, d_f)
    # recompute F --> magic
    fund_matrix = np.dot(u_f, np.dot(d_f_prime, v_f)).T
    print('Should be rank 2: ', np.linalg.matrix_rank(fund_matrix))
    return fund_matrix

if __name__ == '__main__':

    # load images and match files for the first example

    I1 = Image.open('../data/library/library1.jpg')
    I2 = Image.open('../data/library/library1.jpg')
    matches = np.loadtxt('../data/library/library_matches.txt')

    N = len(matches)
    matches_to_plot = copy.deepcopy(matches)

    '''
    Display two images side-by-side with matches
    this code is to help you visualize the matches, you don't need to use it to produce the results for the assignment
    '''

    I3 = np.zeros((I1.size[1], I1.size[0] * 2, 3))
    I3[:, :I1.size[0], :] = I1
    I3[:, I1.size[0]:, :] = I2
    matches_to_plot[:, 2] += I2.size[0]
    I3 = np.uint8(I3)
    I3 = Image.fromarray(I3)
    # fig, ax = plt.subplots()
    # ax.set_aspect('equal')
    # ax.imshow(I3)
    # colors = iter(cm.rainbow(np.linspace(0, 1, matches_to_plot.shape[0])))
    # [plt.plot([m[0], m[2]], [m[1], m[3]], color=next(colors)) for m in matches_to_plot]
    # plt.show()

    # first, fit fundamental matrix to the matches
    F = fit_fundamental_matrix(matches)  # this is a function that you should write
    '''
    display second image with epipolar lines reprojected from the first image
    '''
    M = np.c_[matches[:, 0:2], np.ones((N, 1))].transpose()
    L1 = np.matmul(F, M).transpose()  # transform points from
    # the first image to get epipolar lines in the second image

    # find points on epipolar lines L closest to matches(:,3:4)
    l = np.sqrt(L1[:, 0] ** 2 + L1[:, 1] ** 2)
    L = np.divide(L1, np.kron(np.ones((3, 1)), l).transpose())  # rescale the line
    pt_line_dist = np.multiply(L, np.c_[matches[:, 2:4], np.ones((N, 1))]).sum(axis=1)
    closest_pt = matches[:, 2:4] - np.multiply(L[:, 0:2], np.kron(np.ones((2, 1)), pt_line_dist).transpose())

    # find endpoints of segment on epipolar line (for display purposes)
    pt1 = closest_pt - np.c_[L[:, 1], -L[:, 0]] * 10  # offset from the closest point is 10 pixels
    pt2 = closest_pt + np.c_[L[:, 1], -L[:, 0]] * 10

    # display points and segments of corresponding epipolar lines
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(I2)
    ax.plot(matches[:, 2], matches[:, 3], '+r')
    ax.plot([matches[:, 2], closest_pt[:, 0]], [matches[:, 3], closest_pt[:, 1]], 'r')
    ax.plot([pt1[:, 0], pt2[:, 0]], [pt1[:, 1], pt2[:, 1]], 'g')
    plt.show()
