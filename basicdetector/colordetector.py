#	Imports
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import cv2
#	Reference: https://www.pyimagesearch.com/2016/02/15/determining-object-color-with-opencv/
class ColorDetector:	
	def __init__(self):
		#	TODO: Implement color list importing https://www.rapidtables.com/web/color/RGB_Color.html
		colors = OrderedDict({
			"white": (255,255,255),
			"silver": (191,191,191),
			"gray": (127,127,127),
			"black": (0,0,0),
			"red": (255,0,0),
			"maroon": (127,0,0),
			"orange": (255,127,0),
			"yellow": (255,255,0),
			"olive": (127,127,0),
			"lime": (0,255,0),
			"green": (0,127,0),
			"cyan": (0,255,255),
			"teal": (0,127,127),
			"blue": (0,127,255),
			"indigo": (0,0,255),
			"navy": (0,0,127),
			"magenta": (255,0,255),
			"purple": (127,0,255),
			"brown": (102,51,0),
			"tan": (150,90,60)
		})
		
		self.lab = np.zeros((len(colors), 1, 3), dtype="uint8")
		self.colorNames = []
		
		#	Update the array and the color names list
		for (i, (name, rgb)) in enumerate(colors.items()):
			self.lab[i] = rgb
			self.colorNames.append(name)
 
		#	Convert the array from the RGB color space to LAB color space
		self.lab = cv2.cvtColor(self.lab, cv2.COLOR_RGB2LAB)
		
	def label(self, image, contour):
		#	Create a mask for the contour and compute average color in contour
		mask = np.zeros(image.shape[:2], dtype="uint8")
		cv2.drawContours(mask, [contour], -1, 255, -1)
		mask = cv2.erode(mask, None, iterations=2)
		mean = cv2.mean(image, mask=mask)[:3]
		
		#	Initialize minDist to infinity
		minDist = (np.inf, None)
 
		#	Loop over the LAB color array
		for (i, row) in enumerate(self.lab):
			#	Compute distance between current color in LAB array and average color in contour
			d = dist.euclidean(row[0], mean)
 
			#	Update minDist
			if d < minDist[0]:
				minDist = (d, i)
 
		#	Return name of the closest color
		return self.colorNames[minDist[1]]
