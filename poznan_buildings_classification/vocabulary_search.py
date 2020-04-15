import cv2
import numpy as np

import pickle
from sklearn.cluster import KMeans

from load_dataset import load_dataset


def main():
    # Load dataset
    X, y, y_dictionary = load_dataset('./dataset_train/')

    # Imgs to gray and resize to have the same width
    width = 500
    X = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in X]
    X_norm = [cv2.resize(img, (width, int(img.shape[0]*width/img.shape[1]))) for img in X]

    # Detect features
    feature_detector_descriptor = cv2.AKAZE_create()
    features = []
    for img in X_norm:
        kpt, des = feature_detector_descriptor.detectAndCompute(img, None) 
        features.extend(des)
    features = np.array(features)
    print(features.shape)

    # Create vocabulary
    nb_words = 20
    k_means = KMeans(n_clusters = nb_words)
    k_means.fit(features)
    
    # Save model vocabulary
    vocab_filename = './vocabulary_models/vocabulary_'+str(nb_words)+'words.sav'
    pickle.dump(k_means, open(vocab_filename, 'wb'))


if __name__ == '__main__':
    main()