import json
import matplotlib.pyplot as plt
from numpy import array, zeros
import numpy as np
from scipy.misc import imread, imsave
from glob import glob
from PIL import Image

"""
This code will convert the given masks to regions with neuron coordinates
"""
mask = Image.open('./train_masks/*.png')
# print(mask.toarray())
mask = array(mask)
coordinates = []
print(mask.shape)
print(np.unique(mask))


for x in range(mask.shape[0]):
    for y in range(mask.shape[1]):
        if mask[x, y] != 0:
            #print(mask[x, y])
            coordinates.append([x, y])

imsave('./predictions/', coordinates)
