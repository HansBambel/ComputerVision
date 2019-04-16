import matplotlib.pyplot as plt
import cv2
import numpy as np

path = "Second Week Exercises/exercise1_basics"
pic1name = "lab1a.jpg"
pic1 = plt.imread(f"{path}/{pic1name}")
pic2name = "lab1b.jpg"
pic2 = plt.imread(f"{path}/{pic2name}")

# a) and c)
# fig, axs = plt.subplots(3)
# axs[0].imshow(pic1)
# axs[1].imshow(pic2)
# pic1 = cv2.imread(f"{path}/{pic1name}")
# pic1scaled = cv2.resize(pic1, (100, 100))
# axs[2].imshow(pic1scaled)
# plt.show()

# b) create a matrix and plot it as an image
# fig, axs = plt.subplots(2)
# y = np.linspace(0, 1, 100)
# y = y**2
# A = np.tile(y, (100, 1))
# axs[0].imshow(A)
# axs[1].imshow((A*255).astype(np.uint8))
# plt.show()

# d) convert to grayscale
# pic1grayscale = np.dot(pic1[..., :3], [0.2989, 0.5870, 0.1140])
# pic1grayscale = np.mean(pic1, -1)
# pic1grayscale = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
# plt.imshow(pic1grayscale, cmap=plt.get_cmap('gray'))
# plt.show()

# e) convert to hsv and plot the different channels
# plt.subplot(221)
# plt.imshow(pic1)
# pic1hsv = cv2.cvtColor(pic1, cv2.COLOR_BGR2HSV)
# plt.subplot(222)
# plt.imshow(pic1hsv[:, :, 0])
# plt.subplot(223)
# plt.imshow(pic1hsv[:, :, 1])
# plt.subplot(224)
# plt.imshow(pic1hsv[:, :, 2])
# # print(pic1hsv.shape)
# plt.show()


# f) convert grayscale to binary with different thresholds
def exe1_binary(pic1):
    pic1grayscale = np.mean(pic1, -1)
    plt.subplot(231)
    plt.imshow(pic1grayscale, cmap=plt.get_cmap('gray'))
    plt.subplot(232)
    pic1binary = cv2.threshold(pic1grayscale, 127, 255, cv2.THRESH_BINARY)[1]
    plt.imshow(pic1binary, cmap=plt.get_cmap('gray'))
    plt.subplot(233)
    pic1binary = cv2.threshold(pic1grayscale, 32, 255, cv2.THRESH_BINARY)[1]
    plt.imshow(pic1binary, cmap=plt.get_cmap('gray'))
    plt.subplot(234)
    pic1binary = cv2.threshold(pic1grayscale, 96, 255, cv2.THRESH_BINARY)[1]
    plt.imshow(pic1binary, cmap=plt.get_cmap('gray'))
    plt.subplot(235)
    pic1binary = cv2.threshold(pic1grayscale, 173, 255, cv2.THRESH_BINARY)[1]
    plt.imshow(pic1binary, cmap=plt.get_cmap('gray'))
    plt.subplot(236)
    pic1binary = cv2.threshold(pic1grayscale, 213, 255, cv2.THRESH_BINARY)[1]
    plt.imshow(pic1binary, cmap=plt.get_cmap('gray'))
    plt.show()


# exe1_binary(pic1)

# g) Choose bounding box of image and resize it
def exe1_resize(pic):
    plt.subplot(131)
    plt.imshow(pic)
    plt.subplot(132)
    boundingbox = pic[370:404, 290:370, :]
    plt.imshow(boundingbox)
    plt.subplot(133)
    scale_percent = 220  # percent of original size
    width = int(boundingbox.shape[1] * scale_percent / 100)
    height = int(boundingbox.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    scaledBB = cv2.resize(boundingbox, dim, interpolation=cv2.INTER_CUBIC)
    plt.imshow(scaledBB)
    plt.show()


exe1_resize(pic1)
