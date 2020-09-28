import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import measure, color, io


img = cv2.imread("images/dtg4.png", 0)
pixels_to_um = 0.0000586  # (1 px = 0.0586 nm)

# plt.hist(img.flat, bins = 250, range = (0,255))
ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

kernel = np.ones((3, 3), np.uint8)
eroded = cv2.erode(thresh, kernel, iterations=2)
dilated = cv2.dilate(eroded, kernel, iterations=2)

cv2.imshow("Thresholded Image", thresh)
cv2.imshow("Eroded Image", eroded)
cv2.imshow("Dilated Image", dilated)

mask = dilated == 255

s = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
labeled_mask, num_labels = ndimage.label(mask, structure=s)

img2 = color.label2rgb(labeled_mask, bg_label=0)

cv2.imshow("Colored labels", img2)
cv2.imshow("Original", img)
cv2.waitKey(0)
