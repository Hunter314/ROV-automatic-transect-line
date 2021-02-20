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


def applyHoughTransform(img, edgy_img = None, showall = False, threshold = 100, debug = False):
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
                    pointSlopeFromRhoTheta(rho, theta, debug=False)
                #if showall:
                cv2.line(img, start, end, (0, 0, 255), 2)

            bigLines = bigLinesFromAllLines(lines)
            #print(f'bigLines: {bigLines}')

            for bigLine in bigLines:
                points = pointsFromPointSlope(bigLine["x_intercept"], bigLine["slope"])
                cv2.line(img, points[0], points[1], (0, 255, 0), 2)


            if (len(bigLines) == 2):
                if (bigLines[0]["x_intercept"][0] < bigLines[1]["x_intercept"][0]):
                    # left line is bigLines[0]
                    leftLine = bigLines[0]
                    rightLine = bigLines[1]
                else:
                    leftLine = bigLines[1]
                    rightLine = bigLines[0]
                    # Find desired path
                bisector = x_int_bisector(leftLine, rightLine)
                points = pointsFromPointSlope(bisector["x_intercept"], bisector["slope"])
                #qprint(f"{bisector}\n")
                try:
                    cv2.line(img, points[0], points[1], (255, 255, 0), 2)
                except OverflowError:
                    print(f"First point: {points[0]}\n"
                          f"Second point: {points[1]}\n")
                twist_info = vectorFromBigLines(img_shape=img.shape, line_left=bigLines[0], line_right=bigLines[1])
                cv2.putText(img, text=f"Pitch: {twist_info['rotation']['pitch']:4.2f}  "
                                      f"Yaw: {twist_info['rotation']['yaw']:4.2f}  "
                                      f"Roll: {twist_info['rotation']['roll']:4.2f}  ", org=(0, 60),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                            color=(255, 255, 255), thickness=2)
                cv2.putText(img, text=f"x: {twist_info['translation']['x']:4.2f}  "
                                      f"y: {twist_info['translation']['y']:4.2f}  "
                                      f"z: {twist_info['translation']['z']:4.2f}  ", org=(0, 75),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            color=(255, 255, 255), thickness=2)
            # Now, try to find all major lines in the image
            cv2.putText(img, text=f"Detected {len(bigLines)} lines", org=(0, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

        else:
            cv2.putText(img, text=f"Detected 0 lines", org=(0, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        return lines


def bigLinesFromAllLines(lines, debug=False):
    si_lines = []
    if lines is not None:
        for arr in lines:
            rho = arr[0][0]
            theta = arr[0][1]
            # add line to list of slope intercept lines
            si_lines.append(slopeIntercept(rho, theta))
    else:
        # No big lines if there are no lines!
        return None
    big_lines = []
    line_clusters = []
    intercepts = []
    slopes = []
    thetas = []
    for line in si_lines:

        slopes.append(line[1])
        intercepts.append(line[0][0])
    for lineTheta in lines:
        thetas.append(lineTheta[0][1])
    #plt.plot(intercepts, thetas, 'bo')
    intercepts = np.array(intercepts)
    a = intercepts.reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=3).fit(a)
    # change 600 to image width
    s = np.linspace(0, 600)
    e = kde.score_samples(s.reshape(-1, 1))

    #plt.plot(s, e)
    #plt.show()
    mi, ma = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]
    if(debug):
        print(f"Maxima: {s[ma]}")
        # Now select which lines are close enough to the maxima
        print(f"There are {len(s[ma])} lines")

    for big_line in s[ma]:
        cluster_slopes = []
        cluster_intercepts = []
        for small_line in si_lines:
            if abs(small_line[0][0] - big_line) < 40:
                #add intercept
                cluster_intercepts.append(small_line[0][0])
                cluster_slopes.append(small_line[1])
        if (debug):
            print(f'Cluster slopes: {cluster_slopes}\n Cluster Intercepts: {cluster_intercepts}')
        big_slope = np.average(cluster_slopes)
        big_intercept = np.average(cluster_intercepts)
        big_lines.append({"x_intercept": (big_intercept, 0), "slope" : big_slope})

    return big_lines


def vectorFromBigLines(img_shape, line_left, line_right, debug=False):
    """
    Takes two big lines and returns a direction vector to keep ROV between them.
    :param img_shape: shape of the original image
    :param line_left: left line in point slope form
    :param line_right: right line in point slope form
    :param debug: make True if you're a True fan of print statements
    :return: a tuple of two tuples representing 3d positional and 3d rotational vectors. These vectors describe the ideal movement of the ROV.
    """
    # First, check for rotation of two lines:

    pitch = 0.0
    roll = 0.0
    yaw = 0.0
    x = 0.0
    y = 0.0
    z = 0.0
    bisector = x_int_bisector(line_left, line_right)
    yaw = np.tan(bisector['slope'])

    ideal_margin = img_shape[1] / 6

    l = intersectionWithHorizontal(line_left['x_intercept'], line_left['slope'], horizontal_y = img_shape[0]/2)
    m = intersectionWithHorizontal(line_right['x_intercept'], line_right['slope'], horizontal_y = img_shape[0]/2)
    # Find a line where the slope of the left line is the mirror of the slope of the right line
    print(f'{l} {m}')
    zoom = (m - l) / (ideal_margin * 4)

    x = np.average([l, m]) - 240
    x = zoom * (x / ideal_margin)
    # Ideally:
    # The difference between the blue lines and the edge is 0.25 times the difference between the blue lines
    # 0.25 | 1 | 0.25

    z = 2 * np.log(zoom)

    return {"rotation": {"pitch": pitch, "roll": roll, "yaw": yaw}, "translation": {"x":x, "y": y, "z": z}}

def x_int_bisector(line_a, line_b):
    """
    Returns a line bisecting two lines
    :param line_a: Line in x_intercept slope form
    :param line_b: Line in x_intercept slope form
    :return: the equation of the line bisecting a and b with positive slope.
    """
    try:

        m1 = line_a["slope"]
        m2 = line_b["slope"]
        k1 = line_a['x_intercept'][0]
        k2 = line_b['x_intercept'][0]
    except (KeyError):
        print(f"Could not find bisector. Improper line format. Inputs: {line_a}\n{line_b}\n")
        return None
    # Oh heck, why did I need to use point slope
    d1 = np.sqrt(m1 ** 2 + 1)
    d2 = np.sqrt(m2 ** 2 + 1)
    ans = []
    if (m1 / abs(m1)) * (m2 / abs(m2)) == 1:
        # first case
        denom = d1 + d2
        m = (m1 * d2 + m2 * d1) / denom
        if not (abs(m) <= 10000):
            m = 10000
        y_int_times_denom = m1 * d2 * k1 + m2 * d1 * k2
        x_int = y_int_times_denom / (m1 * d2 + m2 * d1)

    else:
        # second case
        denom = d2 - d1
        m = (m1 * d2 - m2 * d1) / denom
        if not (abs(m) <= 10000):
            m = 10000
        y_int_times_denom = m1 * d2 * k1 - m2 * d1 * k2
        x_int = y_int_times_denom / (m1 * d2 - m2 * d1)


    return {"x_intercept": (x_int, 0), "slope" : m}


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


def getLineColor(img, point, slope):
    # TODO: Finish this to verify two lines on screen are blue
    colors = []
    if (slope >= 1):
        for i in range(0, img.shape[0]):
            colors.append(img[i][i * slope + point])


def pointSlopeFromRhoTheta(rho, theta, debug=False, highVal=1000):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    # Slope is (-b / a)
    # slope is (-np.tan(theta))
    # initial value is (cos(theta)rho)
    if (b != 0):
        slope = -(a / b)
    else:
        # What I'm about to do is kind of yucky, and it's the precise reason why we use rho and theta instead of
        # point slope...
        # but point slope is just easier for many to work in. It's easier for me.
        # For an image that can at most be a couple hundred pixels tall, I feel this value is large enough.
        slope = 100000
        # NOTE: IF you do some kind of fancy yaw calculation based on the far-off intersection point of the two lines,
        # This may mess it up in the case where one line is vertical and the other is not.
        # But like, maybe not that bad. You can always make this number higher.


    if (debug == True):
        print(f'Initial value ({x0}, {y0}), slope {slope}')
    return ((x0, y0), slope)


def pointsFromPointSlope(point, slope, highVal=1000):
    if (abs(slope) <= 1):
        return ((int(point[0] - highVal), int(point[1] - highVal * slope)),
                (int(point[0] + highVal), int(point[1] + highVal * slope)))
    else:
        return ((int(point[0] - highVal / slope), int(point[1] - highVal)),
                (int(point[0] + highVal / slope), int(point[1] + highVal)))


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

