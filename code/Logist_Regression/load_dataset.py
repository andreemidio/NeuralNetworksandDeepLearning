import cv2
import h5py
import numpy as np


class LoadDataSet:
    def load_dataset(self, test, train):
        train_dataset = h5py.File(train)
        test_dataset = h5py.File(test)

        train_set_x_orig = np.array(train_dataset['train_set_x'][:])
        train_set_y_orig = np.array(train_dataset['train_set_y'][:])

        test_set_x_orig = np.array(test_dataset['test_set_x'][:])
        test_set_y_orig = np.array(test_dataset['test_set_y'][:])

        classes = np.array(test_dataset['list_classes'][:])

        train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
        test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

        return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


if __name__ == '__main__':
    test = 'datasets/test_catvnoncat.h5'
    train = 'datasets/train_catvnoncat.h5'

    data = LoadDataSet()

    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = data.load_dataset(test, train)

    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]
    num_px = train_set_x_orig.shape[1]

    print("Number of training examples: m_train = " + str(m_train))
    print("Number of testing examples: m_test = " + str(m_test))
    print("Height/Width of each image: num_px = " + str(num_px))
    print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print("train_set_x shape: " + str(train_set_x_orig.shape))
    print("train_set_y shape: " + str(train_set_y.shape))
    print("test_set_x shape: " + str(test_set_x_orig.shape))
    print("test_set_y shape: " + str(test_set_y.shape))

    index = 24

    cv2.imshow('tet', train_set_x_orig[index])
    cv2.waitKey(0)
    print("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode(
        "utf-8") + "' picture.")
