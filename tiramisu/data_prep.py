import os
import json
from numpy import array, zeros
from scipy.misc import imread, imsave
from glob import glob


def tomask(coords):
    """
    This code is a modified version of the original viewing example.
    Parameters
    ----------
    coords: List
        List of coordinates for the region.
    """

    mask = zeros(dims)
    mask[zip(*coords)] = 1
    return mask


for identities in os.listdir("./project3"):
    if not identities.endswith(".test"):
        id = identities[12:]
        with open("./project3/" + identities + '/regions/regions.json') as f:
            regions = json.load(f)
        for images in os.listdir("./train_data"):
            if not images.startswith(id):
                continue
            dims = [512, 512]
            masks = array([tomask(s['coordinates']) for s in regions])
            imsave('./train_anno/' + images[:16] + '.png', masks.sum(axis=0))
    print(identities)
