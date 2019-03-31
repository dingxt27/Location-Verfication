import numpy as np
import cv2
import random


def chageLight (img):
    image1 = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    random_bright = .25 + np.random.uniform()
    image1[:, :, 2] = image1[:, :, 2] * random_bright
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1


def transform (img):
    ang_range = 25
    ang_rot = np.random.uniform(ang_range) - ang_range / 2
    rows, cols, ch = img.shape
    Rot_M = cv2.getRotationMatrix2D((cols / 2, rows / 2), ang_rot, 1)

    image2 = cv2.warpAffine(img, Rot_M, (cols, rows))

    return image2


def randomGetImage (images,labels,label):
    labels_bins = np.bincount(labels)
    #print(labels_bins)
    FirstIdex = labels.index(label)
    randomIndex = random.randint(0, labels_bins[label]-1)
    selectedImg = images[FirstIdex + randomIndex]

    return selectedImg


def equalize_samples_set(images, labels):
    labels_count_arr = np.bincount(labels)
    labels_bins = np.arange(len(labels_count_arr))

    for label in labels_bins:
        labels_no_to_add = 190 - labels_count_arr[label]

        tempImages = []
        tempLabels = []
        for num in range(labels_no_to_add):
            rand_image = randomGetImage(images, labels, label)
            rand_image = chageLight(rand_image)
            tempImages.append(transform(rand_image))
            tempLabels.append(label)

            #images_temp.append(transform(rand_image))
            #labels_temp.append(label)
        images = images + tempImages
        labels = labels + tempLabels
        #images.append(images_temp)
        #labels.append(labels_temp)

    return images, labels

