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

    # Search vocabulary
    cnt = range(0, 5, 1)
    nb_words = range(5, 50, 5)
    for words in nb_words:
        for i in cnt:
            k_means = KMeans(n_clusters = words)
            k_means.fit(features)
            # Save model vocabulary
            vocab_filename = './vocabulary_models/vocabulary'+str(i)+'_'+str(nb_words)+'words.sav'
            pickle.dump(k_means, open(vocab_filename, 'wb'))
            print('Saved model number '+str(i) + 'with words' +str(words))


if __name__ == '__main__':
    main()