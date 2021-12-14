import cv2
import numpy as np

def get_resistivity_masks(shape, bounds):
    img = cv2.imread('resistivity_mask.png')
    res = cv2.resize(img, dsize=(shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
    mask = np.sum(res, axis=2)
    mask = mask / (256 * 3)
    _min, _max = bounds
    scale = _max - _min
    mask = (mask * scale) + _min  
    return mask, np.copy(mask)

