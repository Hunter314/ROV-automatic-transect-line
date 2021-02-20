import cv2
import numpy as np
import helperFunctions as hp
import matplotlib.pyplot as plt
from datetime import datetime

# Make this True to save the video
# On a side note, are preprocessor macros a thing in Python?
saving = False
#saving = True

now = datetime.now()
date_time = now.strftime("%Y-%m-%d_%H:%M")
save_path = f'output_{date_time}.avi'

cap = cv2.VideoCapture("transectLineExample.mp4")

fps = 24
fourcc = cv2.VideoWriter_fourcc('M', 'P', 'E', 'G')

# Just for the purpose of seeing how many seconds into the video you are in OpenCV
seconds = 0
paused = False

if saving:
    out = cv2.VideoWriter(save_path, fourcc, fps, (480, 360))

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        #print(frame.shape)
        # May need to alter fps depending on video

        if ret and not paused:
            seconds += int(1000 / fps)
            #print(seconds)
            # print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            # print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # All video properties can be found on opencv website
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hframe = frame.copy()
            # phframe = frame.copy()
            # phough not working
            #hboldframe = frame.copy()

            hp.applyHoughTransform(hframe, None, threshold=100, showall=True, debug=True)

            cv2.imshow('Current frame', hframe)
            # cv2.imshow('vales', hp.createValueEdges(frame))
            #hp.applyPHough(phframe, None, minLineLength=50, maxLineGap=40)
            #hp.applyHoughTransform(hframe, hp.boldImage(hp.createHueEdges(frame)), threshold=200)
            #cv2.imshow('Current frame probablistic', phframe)
            #cv2.imshow('Current frame hough with bold', hboldframe)
            # Hough transform test
            if saving:
                out.write(hframe)

        else:
            break

    user_input = cv2.waitKey(int(1000 / fps))
    if user_input is ord('q'):
        break
    if user_input is ord('p'):
        paused = True
    if user_input is ord('r'):
        paused = False

cap.release()
if saving:
    out.release()