#	Imports
import cv2

class FaceDetector:
	def __init__(self):
		pass
		
	def detectFace(self, image, cascade):
		return cascade.detectMultiScale(
			image,
			scaleFactor = 1.1,
			minNeighbors = 5,
			minSize = (30, 30),
			flags = cv2.CASCADE_SCALE_IMAGE
		)