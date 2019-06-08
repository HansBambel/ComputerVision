from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from cyvlfeat import sift as sift_cyvl
import cv2
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
import matplotlib.cm as cm
import copy
from part1_sample_code_python import fit_fundamental_matrix


def find_matching_points(image1, image2, n_levels=3, distance_threshold=300):
    """
    :param image1 and image2 must be RGB images
    :param n_levels: number of scales
    :param distance_threshold: a threshold to accept a given match
    :return: two numpy lists, each with keypoints in [x,y]
    """

    # TODO
    '''
    Important note : you  might need to change the parameters (sift parameters) inside this function to
    have more or better matches
    '''
    matches_1_cyvl = []
    matches_2_cyvl = []
    matches_1_cv = []
    matches_2_cv = []
    image1 = np.array(image1.convert('L'))
    image2 = np.array(image2.convert('L'))
    '''
    Each column of keypoints is a feature frame and has the format [X;Y;S;TH], where X,Y is the (fractional) center of
    the frame, S is the scale and TH is the orientation (in radians).

    AND each column of features is the descriptor of the corresponding frame in F.
    A descriptor is a 128-dimensional vector of class UINT8
    '''
    keypoints_1, features_1 = sift_cyvl.sift(image1, compute_descriptor=True, n_levels=n_levels)
    keypoints_2, features_2 = sift_cyvl.sift(image2, compute_descriptor=True, n_levels=n_levels)
    pairwise_dist = cdist(features_1, features_2)  # len(features_1) * len(features_2)
    closest_1_to_2 = np.argmin(pairwise_dist, axis=1)
    for i, idx in enumerate(closest_1_to_2):
        if pairwise_dist[i, idx] <= distance_threshold:
            matches_1_cyvl.append([keypoints_1[i][1], keypoints_1[i][0]])
            matches_2_cyvl.append([keypoints_2[idx][1], keypoints_2[idx][0]])
    cyvlfeat_matches = np.array(matches_1_cyvl), np.array(matches_2_cyvl)

    sift = cv2.xfeatures2d.SIFT_create(nOctaveLayers=n_levels)
    keypoints_1_cv2, features_1_cv2 = sift.detectAndCompute(image1, None)
    keypoints_2_cv2, features_2_cv2 = sift.detectAndCompute(image2, None)
    pairwise_dist_cv2 = cdist(features_1_cv2, features_2_cv2)  # len(features_1) * len(features_2)
    closest_1_to_2_cv2 = np.argmin(pairwise_dist_cv2, axis=1)
    for i, idx in enumerate(closest_1_to_2_cv2):
        if pairwise_dist_cv2[i, idx] <= distance_threshold:
            matches_1_cv.append([keypoints_1_cv2[i].pt[0], keypoints_1_cv2[i].pt[1]])
            matches_2_cv.append([keypoints_2_cv2[idx].pt[0], keypoints_2_cv2[idx].pt[1]])
    cv2_matches = np.array(matches_1_cv), np.array(matches_2_cv)

    return cyvlfeat_matches
    # return cv2_matches

def calcDist(matches, fund_matrix):
    distances = np.zeros(len(matches))
    for i, m in enumerate(matches):
        p1 = np.array([m[0], m[1], 1]).reshape(3, 1)
        p2 = np.array([m[2], m[3], 1]).reshape(3, 1)
        distances[i] = np.abs(np.dot(p2.T, np.dot(fund_matrix, p1)).squeeze())
    return distances

def RANSAC_for_fundamental_matrix(matches):  # this is a function that you should write
    print('Implementation of RANSAC to to find the best fundamental matrix takes place here')
    # You will iteratively choose some number of point correspondences (8, 9, or some
    # small number), solve for the fundamental matrix using the function you wrote for the
    # part I, and then count the number of inliers. Inliers in this context will be point
    # correspondences that "agree" with the estimated fundamental matrix.
    # In order to count
    # how many inliers a fundamental matrix has, you'll need a distance metric based on the
    # fundamental matrix. (Hint: For a point correspondence (x',x) what properties does the
    # fundamental matrix have?). You'll need to pick a threshold between inlier and outlier
    # and your results are very sensitive to this threshold so explore a range of values.

    # normalize matches?!
    scaler = MinMaxScaler()
    scaler.fit(matches)
    matches_normed = scaler.transform(matches)
    best_fund_matrix = np.eye(3)
    bestError = np.inf
    max_inliers = 0
    s = 9
    N = 5000
    # good for threshold 0.01
    threshold = 0.005
    # e = prob that point is outlier
    e = 0.3
    # T = 200
    T = int((1-e)*len(matches))
    n = 0
    while n < N:
        # get s samples
        samples = np.zeros(len(matches), dtype=bool)
        selected_samples = np.random.choice(range(len(matches)), s, replace=False)
        samples[selected_samples] = True

        # fit a fundamental matrix using the samples
        fund_matrix = fit_fundamental_matrix(matches_normed[samples])
        # check distance to rest of matches
        dist = calcDist(matches_normed[~samples], fund_matrix)
        # print("Mean distance: ", np.mean(dist), "min: ", np.min(dist), "max: ", np.max(dist))
        inliers = matches_normed[~samples][dist < threshold]
        # if amount of Inliers is bigger than T --> good model
        if len(inliers) >= T:
            # fit new matrix with inliers and samples (shape (:,4))
            inliersAndSamples = np.append(matches_normed[samples], inliers, axis=0)
            new_fund_matrix = fit_fundamental_matrix(inliersAndSamples)
            distances = calcDist(matches_normed, new_fund_matrix)
            error = np.sum(distances)
            if error < bestError:
                print(f"Better error in run {n:4d}: {error:3.8f} min: {np.min(distances)} max: {np.max(distances):.3f}")
                best_fund_matrix = new_fund_matrix
                bestError = error
                max_inliers = len(inliersAndSamples)
        n += 1

    # best_fund_matrix, mask = cv2.findFundamentalMat(matches[:, :2], matches[:, 2:4], method=cv2.FM_RANSAC)
    print(f"Threshold: {threshold}, T: {T},  Maximum inliers: {max_inliers} {max_inliers/(len(matches_normed))*100:.2f}%")
    distances = calcDist(matches, best_fund_matrix)
    best_matches = matches[np.argsort(distances)][:100]
    print("Best fundamental Matrix: \n", best_fund_matrix)
    print("Error of all matches: ", np.sum(distances))
    return best_fund_matrix, best_matches

if __name__ == '__main__':
    # load images and match and resize the images
    basewidth = 500
    I1 = Image.open('../data/NotreDame/NotreDame1.jpg')
    wpercent = (basewidth / float(I1.size[0]))
    hsize = int((float(I1.size[1]) * float(wpercent)))
    I1 = I1.resize((basewidth, hsize), Image.ANTIALIAS)

    I2 = Image.open('../data/NotreDame/NotreDame2.jpg')
    wpercent = (basewidth / float(I2.size[0]))
    hsize = int((float(I2.size[1]) * float(wpercent)))
    I2 = I2.resize((basewidth, hsize), Image.ANTIALIAS)

    matchpoints1, matchpoints2 = find_matching_points(I1, I2, n_levels=3, distance_threshold=200)
    matches = np.hstack((matchpoints1, matchpoints2))

    matches_to_plot = copy.deepcopy(matches)

    '''
    Display two images side-by-side with matches
    this code is to help you visualize the matches, you don't need to use it to produce the results for the assignment
    '''

    I3 = np.zeros((I1.size[1], I1.size[0] * 2, 3))
    I3[:, :I1.size[0], :] = I1
    I3[:, I1.size[0]:, :] = I2
    matches_to_plot[:, 2] += I2.size[0]  # add to the x-coordinate of second image
    # fig, ax = plt.subplots()
    # ax.set_aspect('equal')
    # ax.imshow(np.array(I3).astype(int))
    # colors = iter(cm.rainbow(np.linspace(0, 1, matches_to_plot.shape[0])))
    #
    # [plt.plot([m[0], m[2]], [m[1], m[3]], color=next(colors)) for m in matches_to_plot]
    # plt.show()

    # first, find the fundamental matrix to on the unreliable matches using RANSAC
    [F, best_matches] = RANSAC_for_fundamental_matrix(matches)  # this is a function that you should write
    N = len(best_matches)
    '''
    display second image with epipolar lines reprojected from the first image
    '''
    M = np.c_[best_matches[:, 0:2], np.ones((N, 1))].transpose()
    L1 = np.matmul(F, M).transpose()  # transform points from
    # the first image to get epipolar lines in the second image

    # find points on epipolar lines L closest to matches(:,3:4)
    l = np.sqrt(L1[:, 0] ** 2 + L1[:, 1] ** 2)
    L = np.divide(L1, np.kron(np.ones((3, 1)), l).transpose())  # rescale the line
    pt_line_dist = np.multiply(L, np.c_[best_matches[:, 2:4], np.ones((N, 1))]).sum(axis=1)
    closest_pt = best_matches[:, 2:4] - np.multiply(L[:, 0:2], np.kron(np.ones((2, 1)), pt_line_dist).transpose())

    # find endpoints of segment on epipolar line (for display purposes)
    pt1 = closest_pt - np.c_[L[:, 1], -L[:, 0]] * 10  # offset from the closest point is 10 pixels
    pt2 = closest_pt + np.c_[L[:, 1], -L[:, 0]] * 10

    # display points and segments of corresponding epipolar lines
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(I2)
    ax.plot(best_matches[:, 2], best_matches[:, 3], '+r')
    ax.plot([best_matches[:, 2], closest_pt[:, 0]], [best_matches[:, 3], closest_pt[:, 1]], 'r')
    ax.plot([pt1[:, 0], pt2[:, 0]], [pt1[:, 1], pt2[:, 1]], 'g')
    plt.show()

    ## optional, re-estimate the fundamental matrix using the best matches, similar to part1
    # F = fit_fundamental_matrix(best_matches); # this is a function that you wrote for part1
