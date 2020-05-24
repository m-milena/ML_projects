import os
import sys
import cv2
import numpy as np

"""Function for loading dataset from users path
   - load imgs from path, when pictures are gruped in subfolders and based on this subfolders - label them,
   - function return data X (images), data y (labels) and dictionary with labels meaning
"""

def load_dataset(dataset_path):
    data_classes = [f for f in os.listdir(dataset_path) if os.path.isdir(dataset_path + f)]
    X = []
    y = []
    y_dictionary = {}
    for cl in data_classes:
        imgs_names = [f for f in os.listdir(dataset_path+cl) if os.path.isfile(dataset_path+cl+'/'+f)]
        loaded_imgs = np.array([cv2.imread(dataset_path+cl+'/'+img_name, cv2.IMREAD_COLOR) for img_name in imgs_names])
        class_label = data_classes.index(cl)
        labels = np.array([class_label for nb in range(0, loaded_imgs.shape[0])])
        X = np.concatenate((X, loaded_imgs))
        y = np.concatenate((y, labels))
        y_dictionary[class_label] = cl
    return X, y, y_dictionary
