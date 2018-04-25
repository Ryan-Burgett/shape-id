import cv2

class dimension:
	def __init__(self):
		pass
		
	def circleDimensions(self, circle):
		(radius, center) = cv2.minEnclosingCircle(circle)
		perimeter = self.perimeterMeasure(circle)
		print("The radius of this circle is: " + radius + "\n")
		print("The Perimeter of this circle is: " + perimeter + "\n")
	
	def ellipseDimensions(self, ellipse):
		peri = self.perimeterMeasure(ellipse)
		(majRad, center) = cv2.minEnclosingCircle(ellipse)
		print("The perimeter of this ellipse is: " + peri+"\n")
		print("The major radius of this ellipse is: " + majRad + "\n")
	
	def polygonDimensions(self, img, polygon):
		#perimeter = self.perimeterMeasure(polygon)
		perimeter = cv2.arcLength(polygon, True)
		numVerts = len(cv2.approxPolyDP(polygon, 0.01 * perimeter, True))
		corners = cv2.goodFeaturesToTrack(polygon, numVerts, 0.1, 15)
		numLens = numVerts + (numVerts / 2)
		num = int(numVerts/2) + 1
		for i in range(0, num):
			for j in range(i, numVerts):
				#Draw diagonals
				line = cv2.line(img, corners[i],corners[j],(255, 0, 0), 2)
				x = int((corners[i][:0] + corners[j][:0])/2)
				y = int((corners[i][:1] + corners[j][:1])/2)
				t = cv2.arcLength(line, false)
				cv2.putText(img, t, (x,y), 3, 2, (255, 0, 0), 1, LINE_8, false)
		return img

	def perimeterMeasure(shape):
		perimeter = cv2.arcLength(shape, True)
		return perimeter
