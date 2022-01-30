import cv2

img = cv2.imread('ressistivity_mask.png')
res = cv2.resize(img, dsize=(120, 80), interpolation=cv2.INTER_CUBIC)
cv2.imshow('image', res)
cv2.waitKey(0)