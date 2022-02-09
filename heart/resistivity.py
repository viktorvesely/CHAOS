import cv2
import numpy as np

def get_resistivity_masks(shape, bounds, path):
    img = cv2.imread(path)

    if shape is None:
        shape = img.shape

    res = cv2.resize(img, dsize=(shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
    mask = np.sum(res, axis=2)
    mask = mask / (256 * 3)
    _min, _max = bounds
    scale = _max - _min
    mask = (mask * scale) + _min  
    return mask, np.copy(mask)

def show_mask(mask):
    d1, d2 = mask.shape
    img = np.zeros((d1, d2, 3))
    img[:, :,0] = mask
    img[:, :,1] = mask
    img[:, :,2] = mask
    cv2.imshow("hmm", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    mask, _  = get_resistivity_masks((18, 14), (0, 1), "./scared_mask.png")
    print(mask)
    # show_mask(mask)