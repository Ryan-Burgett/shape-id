#	Imports
import cv2
import numpy as np
import math

class CircleDetector:
	def __init__(self):
		pass
	
	def detectCircle(self, contour):
		shape = "unidentified"
		perimeter = cv2.arcLength(contour, True)
		(centerX, centerY), enclosedRad = cv2.minEnclosingCircle(contour)
		
		#	Determine the most extreme points towards the left, right, top, and bottom, along the contour
		extLft = tuple(contour[contour[:, :, 0].argmin()][0])
		extRgt = tuple(contour[contour[:, :, 0].argmax()][0])
		extTop = tuple(contour[contour[:, :, 1].argmin()][0])
		extBot = tuple(contour[contour[:, :, 1].argmax()][0])
		
		distLft = math.sqrt((centerX - extLft[0])**2 + (centerY - extLft[1])**2)
		distRgt = math.sqrt((centerX - extRgt[0])**2 + (centerY - extRgt[1])**2)
		distTop = math.sqrt((centerX - extTop[0])**2 + (centerY - extTop[1])**2)
		distBot = math.sqrt((centerX - extBot[0])**2 + (centerY - extBot[1])**2)
		
		if ((distLft/enclosedRad) <= 1.03 and (distLft/enclosedRad) >= 0.97) and ((distRgt/enclosedRad) <= 1.03 and (distRgt/enclosedRad) >= 0.97) and ((distTop/enclosedRad) <= 1.03 and (distTop/enclosedRad) >= 0.97) and ((distBot/enclosedRad) <= 1.03 and (distBot/enclosedRad) >= 0.97):
			shape = "circle"
		
		return shape;
