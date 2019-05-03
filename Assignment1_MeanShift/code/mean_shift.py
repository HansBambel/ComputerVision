import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
from scipy.spatial.distance import cdist


def findpeak(data, idx, r):
    dist = cdist(data[idx].reshape(1, -1), data)
    # print(data.shape)
    # print((dist <= r).T.reshape(-1).shape)
    return np.mean(data[(dist <= r).T.reshape(-1)], axis=0)


def meanshift(data, r):
    # call findpeak for every point and assign label to point according to peak
    # Note the meanshift algorithm requires that
    # peaks are compared after each call to findpeak and for similar peaks to be merged. For our
    # implementation of meanshift, we will consider two peaks to be the same if the distance between
    # them is smaller than r/2. Also, if the peak of a data point is found to already exist in peaks then for
    # simplicity its computed peak is discarded and it is given the label of the associated peak in peaks.
    data = data.T
    labels = np.zeros(len(data))
    peaks = np.zeros(data.shape)
    t = 0.01
    # repeat until convergence
    for i, point in enumerate(data):
        newPeak = findpeak(data, i, r)
        print(newPeak.shape)
        # get distance from peak to other peaks and maybe merge
        peakDistances = cdist(newPeak.reshape(1, -1), peaks)
        print(peakDistances.shape)
        if peakDistances < r/2:
            pass


    return labels, peaks


points = sio.loadmat("../data/pts.mat")['data']
print("Data shape", points.shape)

labels, peaks = meanshift(points, 2)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.scatter(points[0, :], points[1, :], points[2, :])
plt.show()