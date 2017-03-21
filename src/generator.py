import csv

import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle


# Reads a CSV file in the give folder and stores the lines in an array
# The filenames of the first three columns are adapted to the fit the provided path
# A seperator should be provided for proper parsing of the filenames. Use '/' for Unix and '\\' for Windows.
def read_csv(folder='./data', separator='/'):
    samples = []
    with open(folder + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            for i in range(0, 3):
                path = line[i]
                filename = path.split(separator)[-1]
                line[i] = folder + '/IMG/' + filename
            samples.append(line)
    return samples


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                measurement = float(batch_sample[3])
                correction = 0.2

                center = cv2.imread(batch_sample[0])
                left = cv2.imread(batch_sample[1])
                right = cv2.imread(batch_sample[2])

                measurements.append(measurement)
                images.append(center)

                measurements.append(measurement + correction)
                images.append(left)

                measurements.append(measurement - correction)
                images.append(right)

            augmented_images = []
            augmented_measurements = []
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)

                augmented_images.append(cv2.flip(image, 1))
                augmented_measurements.append(-measurement)

            # trim image to only see section with road
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)
