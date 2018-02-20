#	Imports
from basicdetector.shapedetector import ShapeDetector
from basicdetector.circledetector import CircleDetector
import cv2
import imutils
import argparse
import numpy as np

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

#	Convert the image to gray-scale
grayedImage = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurredImage = cv2.GaussianBlur(grayedImage, (1, 1), 0)
shapeThresh = cv2.threshold(blurredImage, 160, 255, cv2.THRESH_BINARY_INV)[1]

#	Find shape contours in the new optimized image
#	Reference: https://www.pyimagesearch.com/2015/08/10/checking-your-opencv-version-using-python/
contours = cv2.findContours(shapeThresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if imutils.is_cv2():
	contours = contours[0]
else:
	contours = contours[1]
	
shapeD = ShapeDetector()	#	Create our shape detector
circleD = CircleDetector()	#	Create our circle detector

for c in contours:	#	Loop over each contour in contours
	#	Compute the center of the contour
	moment = cv2.moments(c)
	
	#	These must be integers as they correspond to pixel values in our image
	#	Reference: https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html
	centerX = int(resizeRatio * (moment["m10"] / moment["m00"]))
	centerY = int(resizeRatio * (moment["m01"] / moment["m00"]))
	shape = shapeD.detectShape(c)
	
	if shape == "unidentified":
		shape = circleD.detectCircle(c)
	
	#	Resize our contour back to the size of the original image
	c = c.astype("float")
	c *= resizeRatio
	c = c.astype("int")
	
	#	Draw contours, centers, and names on image
	#	Reference: https://www.pyimagesearch.com/2016/02/01/opencv-center-of-contour/
	cv2.drawContours(image, [c], -1, (0,255,0), 2)
	cv2.circle(image, (centerX, centerY), 3, (0,255,0), -1)
	#	Draw text twice, making it stand out better from the background
	cv2.putText(image, shape, (centerX-24, centerY-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
	cv2.putText(image, shape, (centerX-24, centerY-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
		
#	Display the image
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
