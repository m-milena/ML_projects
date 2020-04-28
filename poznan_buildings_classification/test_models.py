import os
import re
import cv2
import pickle
import numpy as np
from load_dataset import load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix

def apply_feature_transform(X, feature_detector_descriptor, vocab_model, nb_words):
    X_transformed = []
    for img in X:
        kpt, des = feature_detector_descriptor.detectAndCompute(img, None) 
        hist = descriptor2histogram(des, vocab_model, nb_words)
        X_transformed.append(hist)
    return X_transformed

def descriptor2histogram(descriptor, vocab_model, nb_words):
    predictions = vocab_model.predict(descriptor)
    hist = np.histogram(predictions, range(0, nb_words+1, 1))
    histogram = hist[0]/len(descriptor)
    return histogram

def main():
    path='./'
    class_model = './models/classificator.sav'
    vocab_model = './models/vocabulary.sav'
    # Info from filename
    name_pattern = r'vocabulary(?P<number>\d(\d)?)_(?P<nb_words>\d(\d)?)words.sav'
    regex = re.compile(name_pattern)
    # Load train, test dataset
    X_train, y_train, y_dictionary = load_dataset('./dataset_train/')
    X_test, y_test, _ = load_dataset('./dataset_test/')
    width = 400
    #X_train = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in X_train]
    X_train = [cv2.resize(img, (width, int(img.shape[0]*width/img.shape[1]))) for img in X_train]
    #X_test = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in X_test]
    X_test = [cv2.resize(img, (width, int(img.shape[0]*width/img.shape[1]))) for img in X_test]
    # Create descriptor
    feature_detector_descriptor =  cv2.xfeatures2d.SIFT_create()#cv2.AKAZE_create()#xfeatures2d.SIFT_create()

    nb = int(regex.search(vocab_model).group('number'))
    nb_words = int(regex.search(vocab_model).group('nb_words'))
    clf_kmeans = pickle.load(open(path+vocab_model, 'rb'))
        # Process dataset
    X_train_trans = apply_feature_transform(X_train, feature_detector_descriptor, clf_kmeans, nb_words)
    X_test_trans = apply_feature_transform(X_test, feature_detector_descriptor, clf_kmeans, nb_words)

    clf_svc = pickle.load(open(path+class_model, 'rb'))
    predicted_train = clf_svc.predict(X_train_trans)
    predicted_test = clf_svc.predict(X_test_trans)
    print(predicted_test)
    acc_train = accuracy_score(predicted_train, y_train)
    acc_test = accuracy_score(predicted_test, y_test) 
    print('Train accuracy: ', acc_train)
    print('Test accuracy: ', acc_test) 
    print(confusion_matrix(y_test, predicted_test))
    for p in range(0, len(predicted_test)):
        print('Predicted: ', y_dictionary.get(predicted_test[p]))
        print('Correct: ', y_dictionary.get(y_test[p]))
        print()
        key = ord('a')
        while key != ord('q'):
            cv2.imshow('d', X_test[p])
            key = cv2.waitKey(3)
        cv2.destroyAllWindows()
            

if __name__ == "__main__":
    main()