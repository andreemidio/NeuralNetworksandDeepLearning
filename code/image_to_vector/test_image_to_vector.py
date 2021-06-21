import numpy as np

from image_to_vector import image_to_vector


def test_image_to_vector():
    image = np.array([[[0.67826139, 0.29380381],
                       [0.90714982, 0.52835647],
                       [0.4215251, 0.45017551]],

                      [[0.92814219, 0.96677647],
                       [0.85304703, 0.52351845],
                       [0.19981397, 0.27417313]],

                      [[0.60659855, 0.00533165],
                       [0.10820313, 0.49978937],
                       [0.34144279, 0.94630077]]])

    result_image = np.array([[0.67826139],
                             [0.29380381],
                             [0.90714982],
                             [0.52835647],
                             [0.4215251],
                             [0.45017551],
                             [0.92814219],
                             [0.96677647],
                             [0.85304703],
                             [0.52351845],
                             [0.19981397],
                             [0.27417313],
                             [0.60659855],
                             [0.00533165],
                             [0.10820313],
                             [0.49978937],
                             [0.34144279],
                             [0.94630077]])

    np.testing.assert_array_equal(image_to_vector(image), result_image)

#
# def test_normalize_rows():
#     x = np.array([[0, 3, 4],
#                   [1, 6, 4]])
#     result = np.array([[0., 0.6, 0.8], [0.13736056, 0.82416338, 0.54944226]])
#     np.testing.assert_array_equal(normalize_rows(x), result)
