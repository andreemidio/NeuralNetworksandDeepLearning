import numpy as np

from load_dataset import LoadDataSet

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
