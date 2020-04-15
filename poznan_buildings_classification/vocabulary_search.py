import cv2
from load_dataset import load_dataset

def main():
    X, y, y_dictionary = load_dataset('./dataset_train/')
    for img in X:
        key = ord('a')
        while key!= ord('q'):
            cv2.imshow('d', img)
            key = cv2.waitKey(3)


if __name__ == '__main__':
    main()