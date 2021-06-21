import numpy as np

from basic_sigmoid import basic_sigmoid, sigmoid


def test_basic_sigmoid():
    assert basic_sigmoid(1) != 1
    assert basic_sigmoid(1) == 0.7310585786300049
    assert basic_sigmoid(2) == 0.8807970779778823
    assert basic_sigmoid(3) == 0.9525741268224334
    assert basic_sigmoid(4) == 0.9820137900379085
    assert basic_sigmoid(5) == 0.9933071490757153
    assert basic_sigmoid(6) == 0.9975273768433653
    assert basic_sigmoid(15) == 0.999999694097773



def test_sigmoid():
    assert sigmoid(1) != 1
    assert sigmoid(0) == 0.5
    assert sigmoid(1) == 0.7310585786300049
    assert sigmoid(2) == 0.8807970779778823
    assert sigmoid(3) == 0.9525741268224334
    assert sigmoid(4) == 0.9820137900379085
    assert sigmoid(5) == 0.9933071490757153
    assert sigmoid(6) == 0.9975273768433653
    assert sigmoid(15) == 0.999999694097773


def test_sigmoid_array_separadamente():
    x = np.array([1, 2, 3])
    assert sigmoid(x)[0] == 0.7310585786300049
    assert sigmoid(x)[1] == 0.8807970779778823
    assert sigmoid(x)[2] == 0.9525741268224334


def test_sigmoid_array():
    x = np.array([1, 2, 3])
    np.testing.assert_array_equal(sigmoid(x), [0.7310585786300049, 0.8807970779778823, 0.9525741268224334])


def test_sigmoid_derivative():
    x = np.array([1, 2, 3])
    np.testing.assert_array_equal(sigmoid(x), [0.7310585786300049, 0.8807970779778823, 0.9525741268224334])
