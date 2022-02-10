import cv2
import numpy as np

class Mask:
    
    def __init__(self, path, grid=None):
        img = cv2.imread(path)

        out=cv2.transpose(img)
        out=cv2.flip(out,flipCode=1)

        self.mask = out

        self.grid = grid

        if grid is None:
            self.grid = self.mask.shape

        self.mask = cv2.resize(
            self.mask,
            dsize=(self.grid[1], self.grid[0]),
            interpolation=cv2.INTER_NEAREST
        ) / 255

    
    def __call__(self, channel, bounds):
        _min, _max = bounds
        scale = _max - _min
        mask = self.mask[:, :, channel]
        mask = (mask * scale) + _min
        return mask  

    def show_channel(self, channel, brightness=1):
        d1, d2, _ = self.mask.shape
        img = np.zeros((d1, d2, 3))
        img[:, :, channel] = np.clip(
            self.mask[:, :, channel] * brightness,
            0, 1
        ) 
        cv2.imshow("hmm", img)
        cv2.waitKey(0)


if __name__ == "__main__":
    import sys

    mask = Mask("./new_mask.png", (70, 200))
    mask.show_channel(int(sys.argv[1]))
    # KK = mask(2, (-14, 38))
    # print(np.max(mask(2, (0, 1))))
    # print(np.min(KK), np.max(KK))

    