#	Imports
from basicdetector.facedetector import FaceDetector
# from imutils.video import VideoStream
# from imutils.video import FPS
import cv2
import imutils
import argparse
import numpy as np
import time

#	Construct an argument parser
ap = argparse.ArgumentParser()
#	Creates our args
ap.add_argument("-i", "--image", required=True, help = "path to the image to be parsed")
ap.add_argument("-c", "--cascade", required=True, help = "path to the face cascade")
ap.add_argument("-e", "--eyes", help = "include this flag to scan faces for eyes", action="store_true")
# ap.add_argument("-v", "--video", help = "include this flag to scan from live video feed", action="store_true")
#	Parse our args
args = vars(ap.parse_args())
#	Process our image
image = cv2.imread(args["image"])
faceCascade = cv2.CascadeClassifier(args["cascade"])
eyeCascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
grayedImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#	Create our FaceDetector and scan image for faces
faceD = FaceDetector()
faces = faceD.detectFace(grayedImage, faceCascade)
#	Draw a rectangle around the faces
for (x, y, w, h) in faces:
	cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), int(w/100) + 2)
	#	If the --eyes flag is included AND the size of the face is not too small,
	#	scan for eyes within the face rectangles
	if args["eyes"] and w > 50:
		eyes = faceD.detectEyes(grayedImage[y:y+h, x:x+w], eyeCascade)
		for (ex, ey, ew, eh) in eyes:
			cv2.rectangle(image[y:y+h, x:x+w], (ex, ey), (ex+ew, ey+eh), (255, 0, 0), int(ew/100) + 1)
#	Display the image
cv2.imshow("Faces found: " + str(len(faces)), image)
cv2.waitKey(0)
cv2.destroyAllWindows()
