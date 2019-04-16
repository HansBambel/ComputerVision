import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

path = "Second Week Exercises/exercise2_edges"
lenapic = plt.imread(f"{path}/Lena.jpg")
# convert to grayscale and normalize
lenaGray = np.mean(lenapic, -1)/255

# plt.imshow(lenaGray, cmap=plt.get_cmap('gray'))
# plt.show()


# b) implement function to get Prewitt and Sobel operator
def operator(name="prewitt"):
    if (name == "prewitt") or (name == "sobel"):
        c = 1 if name == "prewitt" else 2
        horizontal = np.array([[-1, 0, 1], [-1*c, 0, 1*c], [-1, 0, 1]])
        vertical = np.array([[-1, -1*c, -1], [0, 0, 0], [1, 1*c, 1]])
        return horizontal, vertical
    elif name == "laplace":
        return np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    else:
        raise ValueError("Name not known!")


# a) compute gradient magnitude
def thresh(mat, threshold):
    temp = mat
    temp[temp < threshold] = 0
    return temp


def plot_magnitude(pic):
    horizontalFilter, verticalFilter = operator(prewitt=True)
    cHor = my_conv(lenaGray, horizontalFilter)
    cVert = my_conv(lenaGray, verticalFilter)
    # fig, axs = plt.subplots(2)
    # axs[0].imshow(c1, cmap=plt.get_cmap('gray'))
    # axs[1].imshow(c2, cmap=plt.get_cmap('gray'))
    # plt.show()
    grad_magnitude = calc_magnitude(cHor, cVert)

    plt.subplot(231)
    plt.title('Original')
    plt.imshow(pic, cmap=plt.get_cmap('gray'))
    plt.subplot(232)
    plt.title('Gradient Magnitude')
    plt.imshow(grad_magnitude, cmap=plt.get_cmap('gray'))
    plt.subplot(233)
    plt.title('Threshold 0.05')
    plt.imshow(thresh(grad_magnitude, 0.05), cmap=plt.get_cmap('gray'))
    plt.subplot(234)
    plt.title('Threshold 0.25')
    plt.imshow(thresh(grad_magnitude, 0.25), cmap=plt.get_cmap('gray'))
    plt.subplot(235)
    plt.title('Threshold 0.5')
    plt.imshow(thresh(grad_magnitude, 0.5), cmap=plt.get_cmap('gray'))
    plt.subplot(236)
    plt.title('Threshold 0.75')
    plt.imshow(thresh(grad_magnitude, 0.75), cmap=plt.get_cmap('gray'))

    plt.show()


def calc_magnitude(conv1, conv2):
    output = np.sqrt(conv1**2 + conv2**2)
    return output/np.max(output)


def my_conv(pic, filter):
    filterSize = filter.shape[0]//2
    output = np.zeros(pic.shape)
    picPad = np.pad(pic, filterSize, 'constant', constant_values=0)
    # print(picPad)
    for x in range(len(pic)):
        for y in range(len(pic[0])):
            window = picPad[x:x+len(filter), y:y+len(filter)]
            output[x, y] = np.sum(np.multiply(window, filter))
    return output


# plot_magnitude(lenaGray)

# c) apply the Laplacian operator and find zero-crossings
def zeroCrossing(pic):
    output = np.zeros(pic.shape)
    picPad = np.pad(pic, 1, 'constant', constant_values=0)
    for x in range(len(pic)):
        for y in range(len(pic[0])):
            window = picPad[x:x+3, y:y+3]
            # if there is a single difference in signs there is an edge
            output[x, y] = 0 if ((window >= 0).all() or (window < 0).all()) else 1
    return output


plt.subplot(131)
plt.title("Laplace 3")
laplaceFilter3 = operator("laplace")
picLaPl3 = my_conv(lenaGray, laplaceFilter3)
plt.imshow(zeroCrossing(picLaPl3), cmap=plt.get_cmap('gray'))

plt.subplot(132)
plt.title("Laplace 5")
laplaceFilter5 = sio.loadmat(f'{path}/Log5.mat')['Log5']
picLaPl5 = my_conv(lenaGray, laplaceFilter5)
plt.imshow(zeroCrossing(picLaPl5), cmap=plt.get_cmap('gray'))

plt.subplot(133)
plt.title("Laplace 17")
laplaceFilter17 = sio.loadmat(f'{path}/Log17.mat')['Log17']
picLaPl17 = my_conv(lenaGray, laplaceFilter17)
plt.imshow(zeroCrossing(picLaPl17), cmap=plt.get_cmap('gray'))
plt.show()
