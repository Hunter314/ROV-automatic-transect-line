import helperFunctions as hp
import cv2
import numpy as np
from testVideo import test_video

path = "transectLineExample.mp4"

def affine_test(frame):
    # Pick four source points and four destination points
    geom = hp.apply_hough_transform(frame, None, False, 100, True)
    # Use geometry data to find points
    print(geom)


test_video(affine_test, [], path)

    #cv2.getAffineTransform()

