import cv2
import numpy as np
import helperFunctions as hp
import matplotlib.pyplot as plt
cap = cv2.VideoCapture("transectLineExample.mp4")

fourcc = cv2.VideoWriter_fourcc('M', 'P', 'E', 'G')

out = cv2.VideoWriter('output_vid.avi', fourcc, 10, (360, 480))

while cap.isOpened():
    ret, frame = cap.read()
    #print(frame.shape)
    # May need to alter fps depending on video
    fps = 30
    if ret:

        # print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # All video properties can be found on opencv website
        out.write(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hframe = frame.copy()
        # phframe = frame.copy()
        # phough not working
        #hboldframe = frame.copy()

        hp.applyHoughTransform(hframe, None, threshold=100, showall=True)

        cv2.imshow('Current frame', hframe)
        # cv2.imshow('vales', hp.createValueEdges(frame))
        #hp.applyPHough(phframe, None, minLineLength=50, maxLineGap=40)
        #hp.applyHoughTransform(hframe, hp.boldImage(hp.createHueEdges(frame)), threshold=200)
        #cv2.imshow('Current frame probablistic', phframe)
        #cv2.imshow('Current frame hough with bold', hboldframe)
        # Hough transform test



        if cv2.waitKey(int(1000/fps)) is ord('q'):
            break
    else:
        break

cap.release()
out.release()