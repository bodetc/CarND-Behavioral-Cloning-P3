import csv

import cv2


def read_images(folder, separator):
    def get_image(line, index):
        path = line[index]
        filename = path.split(separator)[-1]
        return cv2.imread(folder+'/IMG/'+filename)

    lines = []
    with open('./data/driving_log.csv') as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)
        for line in reader:
            lines.append(line)

    images = []
    measurements = []
    for line in lines:
        measurement = float(line[3])
        correction = 0.5

        center = get_image(line, 0)
        left = get_image(line, 1)
        right = get_image(line, 2)

        measurements.append(measurement)
        images.append(center)

        measurements.append(measurement+correction)
        images.append(left)

        measurements.append(measurement-correction)
        images.append(right)

    return images, measurements


def augment_images(images, measurements):
    augmented_images = []
    augmented_measurements = []
    for image, measurement in zip(images, measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)

        augmented_images.append(cv2.flip(image, 1))
        augmented_measurements.append(-measurement)

    return augmented_images, augmented_measurements
