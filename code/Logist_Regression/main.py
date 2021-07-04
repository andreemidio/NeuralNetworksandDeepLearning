import cv2
import numpy as np
from matplotlib import pyplot as plt

from load_dataset import LoadDataSet
from model import model
from predict import predict

test = 'datasets/test_catvnoncat.h5'

train = 'datasets/train_catvnoncat.h5'

data = LoadDataSet()

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = data.load_dataset(test, train)

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

print('% of Non-cat in the training data: ', 100 * np.sum(train_set_y == 0) / len(train_set_y[0]))
print('% of Cat in the training data: ', 100 * np.sum(train_set_y == 1) / len(train_set_y[0]))

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=20000, learning_rate=0.0005, print_cost=True)

costs = np.squeeze(d['costs'])


def own_Image(my_image):
    # We preprocess the image to fit your algorithm.
    fname = my_image
    image = np.array(cv2.imread(fname))
    my_image = cv2.resize(image, (num_px, num_px))

    my_image = my_image.reshape((1, num_px * num_px * 3)).T

    my_predicted_image = predict(d["w"], d["b"], my_image)

    plt.imshow(image)
    print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[
        int(np.squeeze(my_predicted_image)),].decode("utf-8") + "\" picture.")


own_Image('images/cat.png')
