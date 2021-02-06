import cv2
import numpy as np
import helperFunctions as hp
import matplotlib.pyplot as plt

img = cv2.imread("testImage_15.jpg")
cv2.imshow("Test Image", img)
img_edges = hp.createHueEdges(img)
img_edges_bold = cv2.imshow("Bold", hp.boldImage(img_edges, width=2))
#hp.boldImage(img)
img_phough = img.copy()
hp.applyPHough(img_phough, edgy_img = img_edges_bold, debug = True)
cv2.imshow("Phough", img_phough)
img_hough = img.copy()
lines = hp.applyHoughTransform(img_hough, edgy_img = img_edges, debug = True)
cv2.imshow("hough", img_hough)
if lines is not None:
    print(lines.shape)
    plot1 = plt.figure(1)
    plt.title("Theta")
    plt.hist(lines[:, 0, 1], bins=10)
    plot2 = plt.figure(2)
    plt.title("Rho")
    plt.hist(lines[:, 0, 0], bins=10)

    #plt.show()
cv2.waitKey()