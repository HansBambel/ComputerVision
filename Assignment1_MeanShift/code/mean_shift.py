import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
from scipy.spatial.distance import cdist
from skimage.color import lab2rgb, rgb2lab
import tqdm
import cv2
from time import time
# import plotly
# import plotly.graph_objs as go


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
        # color = np.random.uniform(0, 1, 3)
        #TODO: instead of random color, you can use peaks when you work on actual images
        print(peak)
        color = peak
        cluster = data[np.where(labels == idx)[0]].T
        ax.scatter(cluster[0], cluster[1], cluster[2], c=[color], s=.5)
    plt.show()


def findpeak(data, idx, r, opt=False):
    dist = cdist(data[idx].reshape(1, -1), data)
    # print(data.shape)
    # print((dist <= r).T.reshape(-1).shape)
    peak = np.mean(data[(dist <= r).T.reshape(-1)], axis=0)
    if opt:
        inSearchPath = cdist(peak.reshape(1, -1), data)[0] < r/4
        return peak, inSearchPath
    else:
        return peak, np.zeros(len(data), dtype=bool)


def findpeak_opt(data, idx, r):
    return findpeak(data, idx, r, opt=True)

def meanshift_opt(data, r):
    return meanshift(data, r, speedup1=True)

def meanshift(data, r, speedup1=False, speedup2=False):
    # call findpeak for every point and assign label to point according to peak
    labels = np.zeros(len(data))
    peaks = np.copy(data)
    t = 0.01

    numLabels = 0
    runs = 0
    # repeat until convergence
    converged = False
    while not converged:
        oldPeaks = np.copy(peaks)
        # this array is to skip the peaks that are the same anyway
        toLookAt = np.ones(len(data), dtype=bool)
        for i, point in enumerate(tqdm.tqdm(data)):
            # if the point is already moved with another peak don't touch it again
            if not toLookAt[i]:
                continue
            if speedup2:
                newPeak, inSearchPath = findpeak_opt(data, i, r)
            else:
                newPeak, inSearchPath = findpeak(data, i, r)

            # get distance from peak to other peaks and maybe merge
            peakDistances = cdist(newPeak.reshape(1, -1), peaks)[0]

            samePeaks = peakDistances < r/2
            # If there are peaks in r/2-range --> give same label
            if np.sum(samePeaks) > 0:
                # newPeak is the mean of the peaks in range
                newPeak = np.mean(peaks[samePeaks], axis=0)
                # First occurence in peaks is the new peak for the others (according to assignment)
                # newPeak = peaks[samePeaks][0]
                nonZeroLabels = labels[samePeaks][labels[samePeaks] > 0]
                if len(nonZeroLabels) == 0:
                    numLabels = numLabels+1
                    newLabel = numLabels
                else:
                    newLabel = np.median(nonZeroLabels)
                    # newLabel = labels[samePeaks][0]

                peaks[samePeaks] = newPeak
                labels[samePeaks] = newLabel

            else:
                # Peak is different enough to get assigned a new label
                numLabels += 1
                newLabel = numLabels
            peaks[i] = newPeak
            labels[i] = newLabel

            if speedup1:
                # Speedup 1
                basin = cdist(newPeak.reshape(1, -1), peaks)[0]
                peaks[basin < r] = newPeak
                labels[basin < r] = newLabel
                toLookAt[basin < r] = False
            # Speedup 2 --> set those points to the peak that are in the search path
            peaks[inSearchPath] = newPeak
            labels[inSearchPath] = newLabel
            toLookAt[inSearchPath] = False
            # if newLabel == 0:
            #     print("NewLabel == 0 --> something is off...")
        runs += 1

        peakMovements = np.sum(np.abs(oldPeaks-peaks))
        print(f"Run {runs} with peakmovements: {peakMovements}")
        if peakMovements < t:
            converged = True
        # if runs > 20:
        #     converged = True

    return labels, peaks


def debugData(points):
    labels, peaks = meanshift_opt(points, 2)
    print("Number of unique labels: ", len(np.unique(labels)))
    print(f'Final labels: {np.unique(labels)} and peaks: {peaks}')

    plotclusters3D(points, labels, peaks)

    # nice plot with plotly
    # trace = go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers")
    # layout = go.Layout(title='Cluster')
    # plotly.offline.plot(go.Figure(data=[trace], layout=layout), filename="Points.html", auto_open=True)


def imSegment(image, r, name=None, featureType='3D', speedup1=True, speedup2=True, load=False):
    # convert image to lab
    conv_image = rgb2lab(image)
    print("Image shape: ", conv_image.shape)
    if featureType == '5D':
        # if featureType is 5D add coordinates
        xCoords = np.expand_dims(np.tile(np.array(range(image.shape[0])).reshape(-1, 1), reps=image.shape[1]), axis=-1)
        yCoords = np.expand_dims(np.tile(np.array(range(image.shape[1])).reshape(-1, 1), reps=image.shape[0]).T, axis=-1)
        conv_image = np.append(conv_image, xCoords, axis=-1)
        conv_image = np.append(conv_image, yCoords, axis=-1)
        flattened_image = conv_image.reshape(-1, 5)
    else:
        flattened_image = conv_image.reshape(-1, 3)
    print("Flattened shape: ", flattened_image.shape)

    if load:
        labels = np.load(f"labels_{name}.npy")
        peaks = np.load(f"peaks_{name}.npy")
    else:
        labels, peaks = meanshift(flattened_image, r, speedup1=speedup1, speedup2=speedup2)
    print("peaks.shape ", peaks.shape)
    # Save labels and peaks
    if name is not None:
        np.save(f"labels_{name}", labels)
        np.save(f"peaks_{name}", peaks)

    print("Number of unique labels: ", len(np.unique(labels)))
    # convert segmentation back to rgb

    if featureType == "5D":
        # get rid of coordinates again
        peaks = peaks[:, 0:3]
    segmented_image = peaks.reshape(image.shape)
    segmented_image_rgb = lab2rgb(segmented_image)
    # plotclusters3D(flattened_image, labels, peaks)
    return segmented_image_rgb, labels


# points = sio.loadmat("../data/pts.mat")['data']
# points = points.T
# print("Points shape", points.shape)
# debugData(points)
# imSegment(points, 2, "points")
# plotclusters3D(points, np.load("labels_points.npy"), np.load("peaks_points.npy"))

picture = "face1"
image = plt.imread(f"../data/{picture}.jpg")
load = False

r=10
ft="5D"
speedup1 = True
speedup2 = False
preproccess = "blur"

if preproccess == "blur":
    image_processed = cv2.blur(image, ksize=(3, 3))
elif preproccess == "sharpen":
    sharpening_kernel = np.array([[-1, -1, -1],
                                   [-1, 9, -1],
                                   [-1, -1, -1]])
    # applying the sharpening kernel to the input image & displaying it.
    image_processed = cv2.filter2D(image, -1, sharpening_kernel)
else:
    image_processed = image
plt.subplot(121)
plt.imshow(image)
plt.subplot(122)
plt.imshow(image_processed)
plt.show()

print(f"Image size: {image.shape[:2]}, preprocess={preproccess}, r={r}, featureType={ft}, speedup1={speedup1}, speedup2={speedup2}")

start_time = time()
segmented_image, labels = imSegment(image_processed, r, picture, featureType=ft, speedup1=speedup1, speedup2=speedup2, load=load)
print(f"Took {time()-start_time} seconds")

plt.imshow(segmented_image)
plt.show()
# plotclusters3D(image, labels, segmented_image)