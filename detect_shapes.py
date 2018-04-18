#	Imports
import cv2
import imutils
import argparse
import numpy as np
from basicdetector.shapedetector import ShapeDetector
from basicdetector.circledetector import CircleDetector
from basicdetector.colordetector import ColorDetector

#	Construct an argument parser
ap = argparse.ArgumentParser()
#	Creates arg -i for image and requires it for operation
ap.add_argument("-i", "--image", required=True, help = "path to the image to be parsed")
ap.add_argument("--color", help="include the color of the contour", action="store_true")
ap.add_argument("--area", help="include the area of the contour", action="store_true")
ap.add_argument("--filter", help="Use advanced filtering method for noisy images", action="store_true")
#	Parse our args
args = vars(ap.parse_args())

#	Pre-process our image
image = cv2.imread(args["image"])
height, width, channels = image.shape
#	cv2.imshow("Image", shapeThresh)
#	cv2.waitKey(0)
blurFactor = int(width*.01)
if blurFactor % 2 == 0:
	blurFactor = blurFactor + 1

bilateralImage = cv2.bilateralFilter(image, 2*blurFactor+1, 5*blurFactor, blurFactor)
blurredImage = cv2.GaussianBlur(bilateralImage, (blurFactor, blurFactor), 0)
grayedImage = cv2.cvtColor(blurredImage, cv2.COLOR_BGR2GRAY)
labImage = cv2.cvtColor(blurredImage, cv2.COLOR_BGR2LAB)
shapeThresh = cv2.adaptiveThreshold(grayedImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 2)

if args["filter"]:
	threshx = int(blurFactor/2);
	if threshx % 2 == 0:
		threshx = threshx + 1
	shapeThresh = cv2.adaptiveThreshold(grayedImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, threshx, 2)

#	Everything after this is added and experimental
if args["filter"]:
	blurFactor = int(width*.005)
	if blurFactor % 2 == 0:
		blurFactor = blurFactor + 1

	shapeThresh = cv2.GaussianBlur(shapeThresh, (blurFactor, blurFactor), 0)
	shapeThresh = cv2.adaptiveThreshold(shapeThresh, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 3, 2)
	blurFactor = int(width*.0025)
	kernel = np.ones((blurFactor,blurFactor),np.uint8)
	morphKernel = np.ones((blurFactor*5+1,blurFactor*5+1),np.uint8)
	
	if blurFactor % 2 == 0:
		blurFactor = blurFactor + 1
		
	shapeThresh = cv2.morphologyEx(shapeThresh, cv2.MORPH_CLOSE, kernel)
	blurFactor = int(width*.00125)
	
	if blurFactor % 2 == 0:
		blurFactor = blurFactor + 1
	
	shapeThresh = cv2.medianBlur(shapeThresh, blurFactor)
	shapeThresh = cv2.dilate(shapeThresh, kernel, iterations = 3)
	shapeThresh = cv2.morphologyEx(shapeThresh, cv2.MORPH_CLOSE, morphKernel)
	shapeThresh = cv2.erode(shapeThresh, kernel, iterations = 2)
	
	#	End experimental filtering
else:
	blurFactor = int(width*.0025)
	kernel = np.ones((blurFactor,blurFactor),np.uint8)
	morphKernel = np.ones((blurFactor*5+1,blurFactor*5+1),np.uint8)

#	Find shape contours in the new optimized image
#	Reference: https://www.pyimagesearch.com/2015/08/10/checking-your-opencv-version-using-python/
contours = cv2.findContours(shapeThresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if imutils.is_cv2():
	contours = contours[0]
else:
	contours = contours[1]

if args["filter"]:
	mask = np.ones(shapeThresh.shape[:2], np.uint8) * 255
	thresh = int(width*0.01)
	if thresh % 2 == 0:
		thresh = thresh + 1
	
	for c in contours:
		backup = c
		perimeter = cv2.arcLength(c,True)
		epsilon = 0.1 * perimeter
		area = cv2.contourArea(c)
		if area > thresh * thresh:
			c = cv2.approxPolyDP(c,epsilon,True)
			cv2.drawContours(image, [c], -1, (0,0,255), int(width/500 + 2))
			if cv2.isContourConvex(backup) == True:
				c = cv2.convexHull(backup)
				cv2.drawContours(image, [c], -1, (255,0,255), int(width/500 + 2))
		c = backup
	
	for c in contours:
		shapeThresh = cv2.dilate(shapeThresh, kernel, iterations = 5)
		shapeThresh = cv2.erode(shapeThresh, kernel, iterations = 5)
		area = cv2.contourArea(c)
		if area < thresh * thresh:
			cv2.drawContours(mask, [c], -1, 0, -1)

	masked = cv2.bitwise_and(shapeThresh, shapeThresh, mask=mask)
	
	#Experimental
	cv2.imshow("Image", masked)
	cv2.waitKey(0)
	
	shapeThresh = cv2.GaussianBlur(masked, (blurFactor, blurFactor), 0)
	
	contours = cv2.findContours(shapeThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	if imutils.is_cv2():
		contours = contours[0]
	else:
		contours = contours[1]
	
shapeD = ShapeDetector()	#	Create our shape detector
circleD = CircleDetector()	#	Create our circle detector
if args["color"]:
	colorD = ColorDetector()	#	Create our color detector only if the --color flag is used

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
	
	if args["filter"]:
		backup = c
		epsilon = 0.1 * cv2.arcLength(c,True)
		c = cv2.approxPolyDP(c,epsilon,True)
		shape = shapeD.detectShape(c)
		c = backup
	else:
		shape = shapeD.detectShape(c)
	
	if args["color"]:
		color = colorD.label(labImage, c)	#	Label colors
	
	#	Calculates the area of the contour(in non-null pixels)
	if args["area"]:
		area = cv2.contourArea(c)
		areaText = "area: {}".format(area)
	
	if shape == "unidentified":
		shape = circleD.detectCircle(c)
	
	#	Draw contours, centers, and names on image
	#	Reference: https://www.pyimagesearch.com/2016/02/01/opencv-center-of-contour/
	#	cv2.drawContours(image, [c], -1, (0,255,0), int(width/500 + 2))
	if not args["filter"]:
		cv2.drawContours(image, [c], -1, (0,255,0), 2)
	cv2.circle(image, (centerX, centerY), int(width/500 + 2), (0,255,0), -1)
	#	Determine what text to display
	if args["color"]:
		text = "{} {}".format(color, shape)
	else:
		text = "{}".format(shape)
	#	Draw text twice, making it stand out better from the background
	cv2.putText(image, text, (centerX-24, centerY-8), cv2.FONT_HERSHEY_SIMPLEX, width/2000, (0,0,0), int(width/250 + 3))
	cv2.putText(image, text, (centerX-24, centerY-8), cv2.FONT_HERSHEY_SIMPLEX, width/2000, (255,255,255), int(width/1000 + 1))
	#	Determine if we should display the area
	if args["area"]:
		cv2.putText(image, areaText, (centerX-24, centerY+8), cv2.FONT_HERSHEY_SIMPLEX, width/2000, (0,0,0), int(width/250 + 3))
		cv2.putText(image, areaText, (centerX-24, centerY+8), cv2.FONT_HERSHEY_SIMPLEX, width/2000, (255,255,255), int(width/1000 + 1))
		
#	Display the image
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()