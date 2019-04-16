import numpy as np
import matplotlib.pyplot as plt

path = "Second Week Exercises/exercise3_enhancement-restoration"
pic1name = f'{path}/lab2a.jpg'

pic1 = plt.imread(pic1name)


def contrast_stretching(pic, a, b, ya, yb, alpha, beta, gamma):
    output = np.zeros(pic.shape)
    for x in range(len(pic)):
        for y in range(len(pic[0])):
            if pic[x, y] < a:
                output[x, y] = alpha*pic[x, y]
            elif pic[x, y] >= a and pic[x, y] < b:
                output[x, y] = beta*(pic[x, y] - a) + ya
            else:
                output[x, y] = gamma*(pic[x, y] - b) + yb
    return output


def contrast_clipping(pic, a, b, beta):
    output = np.zeros(pic.shape)
    for x in range(len(pic)):
        for y in range(len(pic[0])):
            if pic[x, y] >= a and pic[x, y] < b:
                output[x, y] = beta*(pic[x, y] - a)
            elif pic[x, y] >= b:
                output[x, y] = beta*(b - a)
    return output


# Part 1a)
# plt.subplot(131)
# plt.title('Original')
# pic1gray = np.mean(pic1, -1)
# plt.imshow(pic1gray, cmap='gray')
#
# plt.subplot(132)
# plt.title('Stretching')
# pic1gray = np.mean(pic1, -1)
# plt.imshow(contrast_stretching(pic1gray, 50, 150, 30, 200, 0.2, 2, 1), cmap=plt.get_cmap('gray'))
#
# plt.subplot(133)
# plt.title('Clipping')
# plt.imshow(contrast_clipping(pic1gray, 50, 200, 1), cmap='gray')
# plt.show()

# 1b)
# pic2 = plt.imread(f'{path}/Unequalized_H.jpg')
# # plt.imshow(pic2, cmap='gray')
# # plt.show()
# h = np.histogram(pic2, bins=range(0, 256))
# # print(h)
#
# pic2equalized = np.zeros(pic2.shape)
# for i in range(0, 256):
#     pic2equalized[pic2 == i] = np.sum(h[0][:i])/np.sum(h[0])*255
# print(pic2equalized)
# print(pic2equalized.shape)
# plt.subplot(121)
# plt.hist(pic2equalized.reshape(-1), bins=range(0, 256))
# plt.subplot(122)
# plt.imshow(pic2equalized, cmap='gray')
# plt.show()

# 1c)
def my_conv(pic, filter):
    filter = filter.T
    filterSize = filter.shape[0]//2
    output = np.zeros(pic.shape)
    picPad = np.pad(pic, filterSize, 'constant', constant_values=0)
    # print(picPad)
    for x in range(len(pic)):
        for y in range(len(pic[0])):
            window = picPad[x:x+len(filter), y:y+len(filter)]
            output[x, y] = np.sum(np.multiply(window, filter))
    return output


blurFilter = np.ones((11, 11))/121
pic1gray = np.mean(pic1, -1)
plt.subplot(121)
plt.imshow(pic1gray, cmap='gray')
plt.subplot(122)
plt.imshow(my_conv(pic1gray, blurFilter), cmap='gray')
plt.show()
