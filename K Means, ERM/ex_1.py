import sys
import math
import numpy as np
import scipy.io.wavfile


def euclideanDistanceCalculation(point, centroid):
    return math.sqrt(((point[0] - centroid[0]) ** 2) + ((point[1] - centroid[1]) ** 2))


def pointToCentroidsCalculation(point, centroids):
    minimumEuclideanDistance = euclideanDistanceCalculation(point, centroids[0])
    indexEuclideanDistance = 0

    for i in range(len(centroids)):
        euclideanDistance = euclideanDistanceCalculation(point, centroids[i])
        if(euclideanDistance < minimumEuclideanDistance):
            minimumEuclideanDistance = euclideanDistance
            indexEuclideanDistance = i

    return indexEuclideanDistance


def calculateMean(points):
    arrayPoints = np.array(points)
    arraySize = len(arrayPoints)

    xCalculation = np.sum(arrayPoints[:, 0]) / arraySize
    yCalculation = np.sum(arrayPoints[:, 1]) / arraySize

    return [round(xCalculation), round(yCalculation)]


def updateCentroids(centroids, centroidDictionary):
    centroidsCopy = centroids.copy()

    for i in range(len(centroids)):
        associatedPointsToCentroid = centroidDictionary[i]
        if len(associatedPointsToCentroid) == 0:
            break
        meanPoint = calculateMean(associatedPointsToCentroid)
        centroidsCopy[i] = meanPoint

    return centroidsCopy


def outputCentroidIterations(outputText, filename):
    file = open(filename, "w")
    file.write(outputText)
    file.close()


def updateDatasetToClosestCentroids(dataset, centroids):
    for i in range(len(dataset)):
        minimumEuclideanDistance = euclideanDistanceCalculation(dataset[i], centroids[0])
        indexEuclideanDistance = 0

        for j in range(len(centroids)):
            distance = euclideanDistanceCalculation(dataset[i], centroids[j])
            if distance < minimumEuclideanDistance:
                minimumEuclideanDistance = distance
                indexEuclideanDistance = j

        dataset[i] = [centroids[indexEuclideanDistance][0], centroids[indexEuclideanDistance][1]]

    return dataset


def main():
    sample, centroids = sys.argv[1], sys.argv[2]

    fs, y = scipy.io.wavfile.read(sample)
    x = np.array(y.copy())
    centroids = np.loadtxt(centroids)

    iterationAmount = 30
    centroidAmount = len(centroids)

    centroidDictionary = dict((i, []) for i in range(centroidAmount))

    outputText = ""
    for iteration in range(iterationAmount):
        print("Iteration number " + str(iteration + 1) + " out of " + str(iterationAmount))

        for point in x:
            closestCentroidIndex = pointToCentroidsCalculation(point, centroids)
            centroidDictionary[closestCentroidIndex].append(point.copy())

        newCentroids = updateCentroids(centroids, centroidDictionary)
        if np.array_equal(centroids, newCentroids) == True:
            print("Converged")
            outputText += "[iter " + str(iteration) + "]:" + ",".join([str(i) for i in newCentroids]) + "\n"
            break

        centroids = newCentroids

        outputText += "[iter " + str(iteration) + "]:" + ",".join([str(i) for i in centroids]) + "\n"
        centroidDictionary = dict((i, []) for i in range(centroidAmount))

    outputCentroidIterations(outputText, "output.txt")
    updatedDataset = updateDatasetToClosestCentroids(x, centroids)
    scipy.io.wavfile.write("compressed.wav", fs, np.array(updatedDataset, dtype=np.int16))


if __name__ == "__main__":
    main()
