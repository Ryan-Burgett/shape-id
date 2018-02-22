#	Imports
import cv2

class FaceDetector:
	def __init__(self):
		pass
		
	def detectFace(self, image, cascade):
		return cascade.detectMultiScale(
			image,
			scaleFactor = 1.15,
			minNeighbors = 10,
			minSize = (30, 30)
		)
		
	def detectEyes(self, image, cascade):
		return cascade.detectMultiScale(image, 1.1, 3)
