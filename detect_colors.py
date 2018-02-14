#	Imports
import cv2
import numpy as np
import imutils
import argparse

#	Construct an argument parser
ap = argparse.ArgumentParser()
#	Creates arg -i for image and requires it for operation
ap.add_argument("-i", "--image", required=True, help = "path to the image to be parsed")
#	Parse our args
args = vars(ap.parse_args())

#	Pre-process our image
#	First, we resize the image with imutils so the shape can be approximated easier
image = cv2.imread(args["image"])
resized = imutils.resize(image, width = 500)
resizeRatio = image.shape[0] / float(resized.shape[0])	#	Keeps track of the ratio of old to new
grayedImage = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurredImage = cv2.medianBlur(grayedImage, 5)

circles = cv2.HoughCircles(blurredImage, cv2.HOUGH_GRADIENT, 1, 260, param1=30, param2=65, minRadius=0, maxRadius=0)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
	cv2.circle(blurredImage,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
	cv2.circle(blurredImage, (i[0],i[1]), 3, (0,255,0), -1)

cv2.imshow('detected circles',blurredImage)
cv2.waitKey(0)
cv2.destroyAllWindows()