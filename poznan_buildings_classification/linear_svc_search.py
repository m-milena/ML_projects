import os
import re
import cv2
import pickle
import numpy as np
from load_dataset import load_dataset

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


tuned_parameters_SVC = [{'kernel': ['linear'], 'C': range(100, 3100, 100)}]


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

def SVC_gridsearch(X_train, X_test, y_train, y_test, nb, nb_words):
    clf_SVC = GridSearchCV(SVC(random_state=42), tuned_parameters_SVC)
    clf_SVC.fit(X_train, y_train)
    predicted = clf_SVC.predict(X_test)
    acc_test = accuracy_score(predicted, y_test) 
    predicted_tr = clf_SVC.predict(X_train)
    acc_train = accuracy_score(predicted_tr, y_train) 
    print("--- Linear SVC best params:  ---")
    print()
    print(clf_SVC.best_params_)
    print()
    print('Accuracy:', acc_train, acc_test)
    print('\n\n')
    # Save model if acc is higher than 80%
    if acc_test >= 0.8:
        filename = './classifiers/svc/svc_acc_' + str(acc_test) + '_acctr_' +  str(acc_train) +'_kmeans_param_' + str(nb) + '_words_' + str(nb_words) + '.sav'
        pickle.dump(clf_SVC, open(filename, 'wb'))

def main():
    folder_path = './vocabulary_models/'
    models_names = [f for f in os.listdir(folder_path) if f[-3:] == 'sav']
    # Info from filename
    name_pattern = r'vocabulary(?P<number>\d(\d)?)_(?P<nb_words>\d(\d)?)words.sav'
    regex = re.compile(name_pattern)
    # Load train, test dataset
    X_train, y_train, y_dictionary = load_dataset('./dataset_train/')
    X_test, y_test, _ = load_dataset('./dataset_test/')
    width = 400
    X_train = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in X_train]
    X_train = [cv2.resize(img, (width, int(img.shape[0]*width/img.shape[1]))) for img in X_train]
    X_test = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in X_test]
    X_test = [cv2.resize(img, (width, int(img.shape[0]*width/img.shape[1]))) for img in X_test]
    # Create descriptor
    feature_detector_descriptor = cv2.xfeatures2d.SIFT_create()

    for model_name in models_names:
        nb = int(regex.search(model_name).group('number'))
        nb_words = int(regex.search(model_name).group('nb_words'))
        clf_kmeans = pickle.load(open(folder_path + model_name, 'rb'))
        # Process dataset
        X_train_trans = apply_feature_transform(X_train, feature_detector_descriptor, clf_kmeans, nb_words)
        X_test_trans = apply_feature_transform(X_test, feature_detector_descriptor, clf_kmeans, nb_words)
        # GridSearch
        SVC_gridsearch(X_train_trans, X_test_trans, y_train, y_test, nb, nb_words)


if __name__ == "__main__":
    main()