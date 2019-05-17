import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler

def nearestNeighbor(k=4):
    pass

def eigenfaces_train(training_images, k=10):
    face_mean = np.mean(training_images, axis=0)
    # plt.imshow(face_mean.reshape(32, 32).T, cmap='gray')
    # plt.show()
    # TODO eigenvectors sohuld have size 1024!!
    face_cov = np.cov(training_images)
    # face_cov = 1/len(training_images) * np.matmul(training_images, training_images.T)
    eigenvals, eigenvecs = np.linalg.eig(face_cov)
    # use the eigenvectors with the biggest eigenvalues (they are sorted already)
    eigenvecsToUse = eigenvecs[:k]
    projectionMatrix = np.dot(eigenvecsToUse, training_images-face_mean)
    return projectionMatrix, face_mean, eigenvecsToUse


numberOfTrainingImages = 3
trainIdx = sio.loadmat(f"../data/{numberOfTrainingImages}Train/{numberOfTrainingImages}.mat")['trainIdx']-1
testIdx = sio.loadmat(f"../data/{numberOfTrainingImages}Train/{numberOfTrainingImages}.mat")['testIdx']-1

print("Train shape", trainIdx.shape)
print("Test shape", testIdx.shape)

data = sio.loadmat("../data/ORL_32x32.mat")
data_gnd = data["gnd"]-1
data_fea = data["fea"]

# print(data)
print(data_gnd.shape)
print(data_fea.shape)
# Normalize data
features = data_fea / 255
print(features)

# plt.subplot(121)
# plt.title("Mean face")
# plt.imshow(np.mean(data_fea, axis=0).reshape(32, 32).T, cmap='gray')
# plt.subplot(122)
# plt.title("Mean face (scaled)")
# plt.imshow(np.mean(features, axis=0).reshape(32, 32).T, cmap='gray')
# plt.show()

trainFaces = np.squeeze(features[trainIdx])
testFaces = np.squeeze(features[testIdx])
# Train with images to get projectionMatrix
k = 6
projectionMatrix, face_mean, eigenvecs = eigenfaces_train(trainFaces, k=k)

plt.title("Eigenfaces")
for i, face in enumerate(projectionMatrix):
    plt.subplot(2, k/2, i+1)
    plt.title(f"{i+1}")
    plt.axis('off')
    plt.imshow(face.reshape(32, 32).T, cmap='gray')
plt.show()

imageToReconstruct = trainFaces[0]
# check whether reconstruction is good
plt.subplot(121)
plt.title("Original Image")
plt.imshow(imageToReconstruct.reshape(32, 32).T, cmap='gray')
plt.subplot(122)
plt.title("Reconstructed Image")
plt.imshow((face_mean + np.dot(eigenvecs[0], projectionMatrix)).reshape(32, 32).T, cmap='gray')
plt.show()
# TODO dont forget to substract the mean of the data
# TODO use euclidean distance for NN and take k neighbors (instead of radius)