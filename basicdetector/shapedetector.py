#	Imports
import cv2

class ShapeDetector:
	def __init__(self):
		pass
		
	def detectShape(self, contour):
		#	Initialize the shape name and approximate the contour
		#	We do this by using OpenCV's cv2.approxPolyDP method with the our contour and the
		#	calculated perimeter.
		shape = "unidentified"
		perimeter = cv2.arcLength(contour, True)
		
		#	OpenCV says that the second parameter passed to the function should be between
		#	one and five percent of the contour perimeter to get the best approximation,
		#	So here I have used one percent. This may be adjusted after testing.
		approximation = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
		
		if len(approximation) == 3: 	#	If the shape has 3 vertices, must be a triangle
			shape = "triangle"
		elif len(approximation) == 4:	#	If the shape has 4 vertices, must be a rectangle
			#	Distinguish between rectangle and square. X&Y are the starting coordinates,
			#	w is width, h is height.
				(x, y, w, h) = cv2.boundingRect(approximation)
				aspectRatio = w / float(h)
				if aspectRatio <= 1.05 and aspectRatio >= 0.95:
					shape = "square"
				else:
					shape = "rectangle"
		elif len(approximation) == 5:	#	If the shape has 5 vertices, must be a pentagon
			shape = "pentagon"
		elif len(approximation) == 6:	#	If the shape has 6 vertices, must be a hexagon
			shape = "hexagon"
		elif len(approximation) == 7:	#	If the shape has 7 vertices, must be a heptagon
			shape = "heptagon"
		elif len(approximation) == 8:	#	If the shape has 8 vertices, must be a octagon
			shape = "octagon"
		
		return shape
