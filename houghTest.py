import helperFunctions as hp
import numpy as np
import cv2

img = cv2.imread("testImage_15.jpg")
#hp.applyHoughTransform(img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# set green and red channels to 0
if False:
    blue = img.copy()
    blue[:, :, 1] = 0
    blue[:, :, 2] = 0
    red = img.copy()
    red[:, :, 0] = 0
    red[:, :, 1] = 0

# create hue img
hue = hsv_img.copy()
hue[:, :, 1] = 255
hue[:, :, 2] = 127



#cv2.imshow("blue,", blue)
#cv2.imshow("red,", red)
cv2.imshow("hue", cv2.cvtColor(hue, cv2.COLOR_HSV2BGR))

#edges = cv2.Canny(gray, 50, 150, apertureSize=3)
edges = cv2.Canny(hue, 100, 150, apertureSize=5)

cv2.imshow("edges", edges)
#blue_edges = cv2.Canny(blue, 50, 150, apertureSize=3)
#cv2.imshow("blue edges,", blue_edges)

lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
#print(lines)
if lines is not None:
    for arr in lines:
        #print("Array at 0:")
        #print(arr)
        rho = arr[0][0]
        theta = arr[0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
cv2.imshow("hough", img)
cv2.waitKey()