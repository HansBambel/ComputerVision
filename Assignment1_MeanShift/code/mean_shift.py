import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
from scipy.spatial.distance import cdist


def plotclusters3D(data, labels, peaks):
    """
    Plots the modes of the given image data in 3D by coloring each pixel
    according to its corresponding peak.

    Args:
        data: image data in the format [number of pixels]x[feature vector].
        labels: a list of labels, one for each pixel.
        peaks: a list of vectors, whose first three components can
        be interpreted as BGR values.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    bgr_peaks = np.array(peaks[:, 0:3], dtype=float)
    rgb_peaks = bgr_peaks[...,::-1]
    rgb_peaks /= 255.0
    for idx, peak in enumerate(rgb_peaks):
        color = np.random.uniform(0, 1, 3)
        #TODO: instead of random color, you can use peaks when you work on actual images
        # color = peak
        cluster = data[np.where(labels == idx)[0]].T
        ax.scatter(cluster[0], cluster[1], cluster[2], c=[color], s=.5)
    plt.show()


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
    # data = data.T
    labels = np.zeros(len(data))
    peaks = np.zeros(data.shape)
    numLabels = 0
    t = 0.01

    runs = 0
    # repeat until convergence
    converged = False

    while not converged:
        for i, point in enumerate(data):
            newPeak = findpeak(data, i, r)
            # print(newPeak.shape)
            # get distance from peak to other peaks and maybe merge
            peakDistances = cdist(newPeak.reshape(1, -1), peaks)[0]
            # print(peakDistances.shape)

            samePeaks = peakDistances < r/2
            # print(samePeaks)
            if np.sum(samePeaks) > 0:
                # print(np.mean(peakDistances[samePeaks], axis=0).shape)
                peaks[samePeaks] = np.mean(peakDistances[samePeaks], axis=0)
                labels[samePeaks] = labels[samePeaks][0]
            else:
                # Peak is different enough to get assigned a new label
                numLabels += 1
                peaks[i] = newPeak
                labels[i] = numLabels
        runs += 1

        print(runs)
        if runs > 20:
            converged = True

    return labels, peaks


points = sio.loadmat("../data/pts.mat")['data']
print("Data shape", points.shape)
points = points.T

# print(points[:, 0], np.min(points[:, 0]), np.max(points[:, 0]))
# print(points[:, 1], np.min(points[:, 1]), np.max(points[:, 1]))
# print(points[:, 2], np.min(points[:, 2]), np.max(points[:, 2]))
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(points[:, 0], points[:, 1], points[:, 2])
# plt.show()

labels, peaks = meanshift(points, 2)

print(f'Final labels: {labels} and peaks: {peaks}')
plotclusters3D(points, labels, peaks)
