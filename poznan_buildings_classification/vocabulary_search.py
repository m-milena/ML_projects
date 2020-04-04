import numpy as np
import cv2
from load_dataset import load_dataset
from sklearn.cluster import KMeans

def main():
    X, y, labels_dict = load_dataset('./dataset/')
    feat_detect = cv2.AKAZE_create()
    features = []
    for img in X:
        kpt, des = feat_detect.detectAndCompute(img, None)
        features.append(des)

if __name__ == '__main__':
    main()