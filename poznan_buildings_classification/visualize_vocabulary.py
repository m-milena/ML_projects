import pickle
import seaborn as sns
import numpy as np
import cv2
import pandas as pd
from load_dataset import load_dataset
import matplotlib.pyplot as plt

def descriptor2histogram(descriptor, vocab_model, nb_words):
    predictions = vocab_model.predict(descriptor)
    hist = np.histogram(predictions, range(0, nb_words+1, 1))
    histogram = hist[0]/len(descriptor)
    return predictions, histogram

def main():
    # Load model
    model_file = './vocabulary_models/vocabulary_20words.sav'
    kmeans_model = pickle.load(open(model_file, 'rb'))
    # Load dataset
    X, y, y_dictionary = load_dataset('./dataset_train/')
    # Imgs to gray and resize to have the same width
    width = 500
    X_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in X]
    X_norm_gray = [cv2.resize(img, (width, int(img.shape[0]*width/img.shape[1]))) for img in X_gray]
    X_norm_color = [cv2.resize(img, (width, int(img.shape[0]*width/img.shape[1]))) for img in X]

    # Detect features
    feature_detector_descriptor = cv2.AKAZE_create()
    features = []
    for img in X_norm_gray:
        kpt, des = feature_detector_descriptor.detectAndCompute(img, None) 
        features.append(des)

    # Histogram
    predictions, hist= descriptor2histogram(features[-1], kmeans_model, 20)
    print(hist)
    sns.set(color_codes=True)
    sns.distplot(predictions, bins=20, kde=True, rug=False)
    plt.xlabel('Words')
    plt.ylabel('Percent of words in img')
    plt.xlim([0, 20])
    plt.xticks(np.arange(0, 20, 1))
    plt.show()
    

    # Features for all imgs
if __name__ == '__main__':
    main()