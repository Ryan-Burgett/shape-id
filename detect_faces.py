#	Imports
from basicdetector.facedetector import FaceDetector
import cv2
import imutils
import argparse
import numpy as np

#	Construct an argument parser
ap = argparse.ArgumentParser()
#	Creates args -i and -c and requires them for operation
ap.add_argument("-i", "--image", required=True, help = "path to the image to be parsed")
ap.add_argument("-c", "--cascade", required=True, help = "path to the cascade to use")
#	Parse our args
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
faceCascade = cv2.CascadeClassifier(args["cascade"])
#grayedImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) TODO: FIX
faceD = FaceDetector()
faces = faceD.detectFace(image, faceCascade)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
	cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
	cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 255), 1)

#	Display the image
cv2.imshow("Faces found", image)
cv2.waitKey(0)
cv2.destroyAllWindows()