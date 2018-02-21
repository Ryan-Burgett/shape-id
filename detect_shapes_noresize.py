#	Imports
from basicdetector.shapedetector import ShapeDetector
from basicdetector.circledetector import CircleDetector
from basicdetector.colordetector import ColorDetector
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
image = cv2.imread(args["image"])
blurredImage = cv2.GaussianBlur(image, (1, 1), 0)
grayedImage = cv2.cvtColor(blurredImage, cv2.COLOR_BGR2GRAY)
labImage = cv2.cvtColor(blurredImage, cv2.COLOR_BGR2LAB)
shapeThresh = cv2.adaptiveThreshold(grayedImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 2)

#	Find shape contours in the new optimized image
#	Reference: https://www.pyimagesearch.com/2015/08/10/checking-your-opencv-version-using-python/
contours = cv2.findContours(shapeThresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if imutils.is_cv2():
	contours = contours[0]
else:
	contours = contours[1]
	
shapeD = ShapeDetector()	#	Create our shape detector
circleD = CircleDetector()	#	Create our circle detector
colorD = ColorDetector()	#	Create our color detector

for c in contours:	#	Loop over each contour in contours
	#	Compute the center of the contour
	moment = cv2.moments(c)
	
	#	These must be integers as they correspond to pixel values in our image
	#	Reference: https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html
	if moment["m00"] != 0:
		centerX = int(moment["m10"] / moment["m00"])
		centerY = int(moment["m01"] / moment["m00"])
	else:
		centerX, centerY = 0, 0
	
	shape = shapeD.detectShape(c)
	color = colorD.label(labImage, c)
	
	if shape == "unidentified":
		shape = circleD.detectCircle(c)
	
	#	Draw contours, centers, and names on image
	#	Reference: https://www.pyimagesearch.com/2016/02/01/opencv-center-of-contour/
	cv2.drawContours(image, [c], -1, (0,255,0), 2)
	cv2.circle(image, (centerX, centerY), 3, (0,255,0), -1)
	#	Draw text twice, making it stand out better from the background
	text = "{} {}".format(color, shape)
	cv2.putText(image, text, (centerX-24, centerY-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
	cv2.putText(image, text, (centerX-24, centerY-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
	
#	Display the image
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
