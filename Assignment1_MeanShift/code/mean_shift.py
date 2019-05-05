import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
from scipy.spatial.distance import cdist
import plotly
import plotly.graph_objs as go


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
    peaks = data
    # labels = np.zeros(data.shape[1]).reshape(1, -1)
    # peaks = np.zeros(data.shape[1]).reshape(1, -1)
    numLabels = 0
    t = 0.01

    numLabels = 0
    runs = 0
    # repeat until convergence
    converged = False
    while not converged:
        oldPeaks = np.copy(peaks)
        for i, point in enumerate(data):
            newPeak = findpeak(data, i, r)
            # print(newPeak.shape)
            # get distance from peak to other peaks and maybe merge
            peakDistances = cdist(newPeak.reshape(1, -1), peaks)[0]
            # print(peakDistances.shape)
            # print(peakDistances)

            samePeaks = peakDistances < r/2
            # print(samePeaks)
            # If there are peaks in r/2-range --> give same label
            if np.sum(samePeaks) > 0:
                # print("Number of same peaks: ", np.sum(samePeaks))
                # print("New Peak: ", newPeak)
                newPeak = np.mean(peaks[samePeaks], axis=0)
                # print("Mean Peak: ", newPeak)
                nonZeroLabels = labels[samePeaks][labels[samePeaks] > 0]
                if len(nonZeroLabels) == 0:
                    numLabels = numLabels+1
                    newLabel = numLabels
                else:
                    newLabel = np.median(nonZeroLabels)

                peaks[samePeaks] = newPeak
                labels[samePeaks] = newLabel

            else:
                # Peak is different enough to get assigned a new label
                numLabels += 1
                newLabel = numLabels
            peaks[i] = newPeak
            labels[i] = newLabel
            if newLabel == 0:
                print("NewLabel == 0 --> something is off...")
        runs += 1

        peakMovements = np.sum(np.abs(oldPeaks-peaks))
        print(f"Run {runs} with peakmovements: {peakMovements}")
        if peakMovements < t:
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
print("Number of unique labels: ", len(np.unique(labels)))
print(f'Final labels: {np.unique(labels)} and peaks: {peaks}')

plotclusters3D(points, labels, peaks)

# nice plot with plotly
# trace = go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers")
# layout = go.Layout(title='Cluster')
# plotly.offline.plot(go.Figure(data=[trace], layout=layout), filename="Points.html", auto_open=True)
