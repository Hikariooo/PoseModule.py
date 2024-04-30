import cv2
import numpy as np
import time

cap = cv2.VideoCapture("1.mp4")
img = cv2.imread("Pose.png")

while True:
    #success, img = cap.read()
    #img = cv2.resize(img, (1280,720))
    
    cv2.imshow("Image", img)
    cv2.waitKey(15)
    