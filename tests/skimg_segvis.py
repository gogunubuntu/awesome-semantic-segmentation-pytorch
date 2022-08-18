#!/usr/bin/env python3

from skimage import io
from skimage import color
from skimage import segmentation
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

# URL for tiger image from Berkeley Segmentation Data Set BSDS
url = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300/html/images/plain/normal/color/108073.jpg"

# Load tiger image from URL
tiger = io.imread(url)

# Segment image with SLIC - Simple Linear Iterative Clustering
seg = segmentation.slic(
    tiger, n_segments=30, compactness=40.0, enforce_connectivity=True, sigma=3
)
print(type(seg))
# Generate automatic colouring from classification labels
overlap = (np.array(color.label2rgb(seg, tiger)) * 255).astype(np.uint8)
cv.imshow("asdf", overlap)
cv.waitKey(0)
