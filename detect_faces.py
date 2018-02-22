#	Imports
from basicdetector.facedetector import FaceDetector
import cv2
import imutils
import argparse
import numpy as np

#	Construct an argument parser
ap = argparse.ArgumentParser()
#	Creates args -i and requires it for operation
ap.add_argument("-i", "--image", required=True, help = "path to the image to be parsed")
#	Parse our args
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
faceCascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
grayedImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faceD = FaceDetector()
faces = faceD.detectFace(grayedImage, faceCascade)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
	cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), int(w/100) + 2)
	eyes = faceD.detectEyes(grayedImage[y:y+h, x:x+w], eyeCascade)
	for (ex, ey, ew, eh) in eyes:
		cv2.rectangle(image[y:y+h, x:x+w], (ex, ey), (ex+ew, ey+eh), (255, 0, 0), int(ew/100) + 1)

#	Display the image
cv2.imshow("Faces found", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
