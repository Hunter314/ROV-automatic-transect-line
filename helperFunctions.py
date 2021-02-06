import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema

def saveTestImageAtSec(cap, sec):
    # do this for the right index
    cap.set(cv2.CAP_PROP_POS_MSEC, 1000 * sec)
    ret, frame = cap.read()
    #fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    #out = cv2.VideoWriter('output.avi', fourcc, 10.0, (640, 480))
    #out.write(frame)
    if (ret):
        cv2.imwrite("testImage_" + str(sec) + ".jpg", frame)
    else:
        return False

#test_cap = cv2.VideoCapture("transectLineExample.mp4")
#saveTestImageAtSec(test_cap, 15)
# max line gap up
# accumulator down


def createHueEdges(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = hsv_img.copy()
    hue[:, :, 1] = 255
    hue[:, :, 2] = 127
    cv2.imshow('Current frame hue', cv2.cvtColor(hue, cv2.COLOR_HSV2BGR))

    edges = cv2.Canny(hue, 100, 150, apertureSize=3)
    cv2.imshow('edges', edges)
    return edges


def createValueEdges(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = hsv_img.copy()
    # remove all saturation
    hue[:, :, 1] = 0
    cv2.imshow('Current frame value', cv2.cvtColor(hue, cv2.COLOR_HSV2BGR))

    edges = cv2.Canny(hue, 100, 150, apertureSize=3)
    cv2.imshow('edges', edges)
    return edges


def boldImage(img, bold_color = 255, width = 1):
    '''Extends every pixel of bold_color to the surrounding width pixels.'''
    copy_img = img.copy()
    arr_img = np.array(img)
    # print(arr_img.shape)
    for i in range(width, arr_img.shape[0] - width):
        for j in range(width, arr_img.shape[1] - width):
            pixel = arr_img[i][j]
            if pixel == bold_color:
                # print("Found a white pixel at " + str(i) + ", " + str(j))
                for k in range(j - width, j + width):
                    for l in range (i - width, i + width):
                        copy_img[l][k] = bold_color
                # copy_img = cv2.rectangle(copy_img, (j - width, i - width), (j + width, i + width), 255, -1)

    return copy_img


def applyHoughTransform(img, edgy_img = None, threshold = 100, debug = False):
    lines = None
    if edgy_img is None:
        applyHoughTransform(img, createHueEdges(img))
    else:
        edges = edgy_img
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)
        # list of slope intercept lines
        si_lines = []
        if (debug):
            print("Hough transform lines:\n" + str(lines) + "End.\n")
        if lines is not None:
            for arr in lines:
                #print("Array at 0:")
                #print(arr)
                rho = arr[0][0]
                theta = arr[0][1]
                start, end = pointsFromRhoTheta(rho, theta)
                if (debug == True):
                    pointSlopeFromRhoTheta(rho, theta, debug=True)
                cv2.line(img, start, end, (0, 0, 255), 2)

                # add line to list of slope intercept lines
                si_lines.append(slopeIntercept(rho, theta))


            big_lines = []
            intercepts = []
            slopes = []
            thetas = []
            for line in si_lines:
                slopes.append(line[1])
                intercepts.append(line[0][0])
            for lineTheta in lines:
                thetas.append(lineTheta[0][1])
            plt.plot(intercepts, thetas, 'bo')
            intercepts = np.array(intercepts)
            a = intercepts.reshape(-1, 1)
            kde = KernelDensity(kernel='gaussian', bandwidth=3).fit(a)
            s = np.linspace(0, 400)
            e = kde.score_samples(s.reshape(-1, 1))

            plt.plot(s, e)
            plt.show()

            # Now, try to find all major lines in the image
            cv2.putText(img, text=f"Detected {len(lines)} lines", org=(0, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        else:
            cv2.putText(img, text=f"Detected 0 lines", org=(0, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

    return lines

def pointsFromRhoTheta(rho, theta, debug=False, highVal=1000):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    # Slope is (-b / a)
    # slope is (-np.tan(theta))
    # initial value is (cos(theta)rho)
    x1 = int(x0 + highVal * (-b))
    y1 = int(y0 + highVal * (a))
    x2 = int(x0 - highVal * (-b))
    y2 = int(y0 - highVal * (a))
    if debug:
        print(f'Initial value ({x0}, {y0}) generates ({x1}, {y1}) to ({x2},{y2})')
    return ((x1, y1), (x2, y2))


def pointSlopeFromRhoTheta(rho, theta, debug=False, highVal=1000):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    # Slope is (-b / a)
    # slope is (-np.tan(theta))
    # initial value is (cos(theta)rho)
    slope = -(a / b)
    if (debug == True):
        print(f'Initial value ({x0}, {y0}), slope {slope}')
    return ((x0, y0), slope)


def slopeIntercept(rho, theta, debug=False):
    """ Returns a line in the form of a slope and its x-intercept.

    For our purposes, the x-intercept makes more sense than the y intercept.
    """
    point, slope = pointSlopeFromRhoTheta(rho, theta)
    return ((intersectionWithHorizontal(point, slope), 0), slope)


def intersectionWithHorizontal(point, slope, horizontal_y=0):
    """ Returns the value x at which the given line (in point slope form) intersects with a horizontal line"""
    x0 = point[0]
    y0 = point[1]
    # a scalar k * slope will separate x0 from x1
    k = y0 - horizontal_y
    # k = dy
    # slope = dy / dx
    # dx = dy / slope
    intersection = x0 + k / slope
    return intersection





def getAllSides(img, edgy_img=None):
    # Will return every rail found in the image of the transect
    applyHoughTransform()



def applyPHough(img, edgy_img = None, minLineLength = 100, maxLineGap = 50, debug = False):
    if edgy_img is None:
        edgy_img = createHueEdges(img)
        applyPHough(img, edgy_img, minLineLength, maxLineGap)
    else:
        edges = edgy_img
        # cv2.imshow("p input", edges)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength, maxLineGap)
        if debug:
            print("Probablistic hough transform lines:\n" + str(lines) + "End.\n")

        if lines is not None:

            for arr in lines:
                x1 = arr[0][0]
                y1 = arr[0][1]
                x2 = arr[0][2]
                y2 = arr[0][3]
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            print("Lines is none")

