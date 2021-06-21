import numpy as np


def image_to_vector(image):
	calc = image.shape[0] * image.shape[1] * image.shape[2]
	v = image.reshape(calc, 1)

	return v







