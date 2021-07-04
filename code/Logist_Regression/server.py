import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy import ndimage

from main import num_px

from predict import predict


def own_Image(my_image):
    # We preprocess the image to fit your algorithm.
    fname = my_image
    image = np.array(ndimage.imread(fname, flatten=False))
    my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
    my_predicted_image = predict(d["w"], d["b"], my_image)

    plt.imshow(image)
    print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")