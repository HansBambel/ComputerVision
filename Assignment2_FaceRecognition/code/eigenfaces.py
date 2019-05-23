import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.spatial.distance import cdist


class NearestNeighbor():

    def train(self, train_images, train_class, eigenvectors, mean_face):
        self.mean_face = mean_face
        self.eigenvectors = eigenvectors
        self.feature_space = np.dot((train_images-self.mean_face), self.eigenvectors.T)
        self.classes = train_class

    def test(self, test_image, k=1):
        # project into feature space
        space = np.dot((test_image-self.mean_face), self.eigenvectors.T).reshape(1, -1)
        distances = cdist(space, self.feature_space)[0]
        closest_neighbors = np.argsort(distances)
        # plt.suptitle("Test")
        # plt.subplot(121)
        # plt.title("Original")
        # plt.imshow(test_image.reshape(32, 32).T, cmap='gray')
        # plt.subplot(122)
        # plt.title("Reconstruction")
        # plt.imshow((face_mean + np.sum(self.eigenvectors * (test_image - self.mean_face), axis=0)).reshape(32, 32).T, cmap='gray')
        # plt.show()
        # returns class
        return np.median(self.classes[closest_neighbors][:k])


def get_best_k(images, k_values):
    accuracies = np.zeros(len(k_values))
    for i, k in enumerate(k_values):
        face_mean, eigenvecs = eigenfaces_train(images, k=k)
        nn = NearestNeighbor()
        nn.train(trainFaces, trainClass, eigenvecs, face_mean)
        # Classify the test images with the trained NN
        classification = np.zeros(len(testClass), dtype=int)
        for j, face in enumerate(testFaces):
            classification[j] = nn.test(face, k=1)
        # calculate accuracy --> correct classifications / all classifications
        accuracies[i] = np.sum(classification == testClass) / len(testClass) * 100
        # errors.append(getConstructionError(images, face_mean, eigenvecs))
    # errors = np.array(errors)
    best_k = k_values[np.argmax(accuracies)]
    return best_k, accuracies


def getConstructionError(images, face_mean, eigenvectors):
    errors = []
    for im in images:
        reconstructed = face_mean + np.sum(eigenvectors * (im - face_mean), axis=0)
        errors.append(np.sum(np.abs(im - reconstructed)))
    # allInOne = np.mean(np.sum(np.abs(images - (face_mean + np.sum(eigenvectors * (images - face_mean), axis=0)))))
    return np.mean(errors)


def reconstruct_images(images, face_mean, eigenvectors):
    plt.suptitle("Reconstruction of images")
    for i, image in enumerate(images):
        plt.subplot(len(images), 2, 2*i+1)
        plt.title("Original")
        plt.axis('off')
        plt.imshow(image.reshape(32, 32).T, cmap='gray')
        plt.subplot(len(images), 2, 2*i+2)
        plt.title("Reconstruction")
        plt.axis('off')
        reconstructed = face_mean + np.sum(eigenvectors * (image - face_mean), axis=0)
        plt.imshow(reconstructed.reshape(32, 32).T, cmap='gray')
    plt.show()


def eigenfaces_train(training_images, k=10):
    face_mean = np.mean(training_images, axis=0)
    face_cov = np.cov(training_images.T)
    eigenvals, eigenvecs = np.linalg.eig(face_cov)
    return face_mean, eigenvecs[:k]

def plot_eigenfaces(training_images, k):
    face_mean = np.mean(training_images, axis=0)
    face_cov = np.cov(training_images)
    _, eigenvectors = np.linalg.eig(face_cov)
    eigenvecsToUse = np.dot(eigenvectors, training_images)[:k]

    plt.suptitle("Eigenfaces")
    for i, face in enumerate(eigenvecsToUse):
        plt.subplot(2, (k+1)//2, i+1)
        plt.title(f"{i+1}")
        plt.axis('off')
        plt.imshow(face.reshape(32, 32).T, cmap='gray')
    plt.show()

    reconstruct_images(training_images[np.random.randint(0, len(training_images), size=5)], face_mean, eigenvecsToUse)


data = sio.loadmat("../data/ORL_32x32.mat")
data_gnd = data["gnd"]
data_fea = data["fea"]
numberOfTrainingImages = 7
# Note: indices are done for Matlab (aRraYs stArT aT 1...) so we need to subtract 1
trainIdx = sio.loadmat(f"../data/{numberOfTrainingImages}Train/{numberOfTrainingImages}.mat")['trainIdx']-1
testIdx = sio.loadmat(f"../data/{numberOfTrainingImages}Train/{numberOfTrainingImages}.mat")['testIdx']-1

# Inspect given data
print("Data label shape ", data_gnd.shape)
print("Data feature shape ", data_fea.shape)
print("Train shape", trainIdx.shape)
print("Test shape", testIdx.shape)
# Normalize data (since grayscale --> divide by 255)
features = data_fea / 255

# Split the data into train and test set
trainFaces = np.squeeze(features[trainIdx])
testFaces = np.squeeze(features[testIdx])
trainClass = np.squeeze(data_gnd[trainIdx])
testClass = np.squeeze(data_gnd[testIdx])

# Train with images to get eigenvectors
# find best k
k_values = np.arange(50)+1
best_k, accuracies = get_best_k(trainFaces, k_values)
print(f"Highest Accuracy of {accuracies[np.argmax(accuracies)]:.2f}% with k={best_k}")

plt.plot(k_values, accuracies)
plt.xlabel("Number of eigenvectors")
plt.ylabel("Accuracy")
plt.show()

# best_k = 80
face_mean, eigenvecs = eigenfaces_train(trainFaces, best_k)
# Plot the eigenfaces
plot_eigenfaces(trainFaces, best_k)

# reconstruct a few images
# reconstruct_images(trainFaces[:5], face_mean, eigenvecs)

# use euclidean distance for NN and take k neighbors (instead of radius)
# Train Nearest neighbor
nn = NearestNeighbor()
nn.train(trainFaces, trainClass, eigenvecs, face_mean)
# Classify the test images with the trained NN
classification = np.zeros(len(testClass), dtype=int)
for i, face in enumerate(testFaces):
    classification[i] = nn.test(face, k=1)
# calculate accuracy --> correct classifications / all classifications
accuracy = np.sum(classification==testClass)/len(testClass)*100
print(f'Accuracy: {accuracy:.2f}%')

