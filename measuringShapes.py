import cv2
import numpy as np
import math

#if shape = circle, call this
#parameters:
	#the contour for the shape.
	#the image that it comes from
	#the number of vertexes--polygons only
class dimension:
#How want these to be given back to the user?
	def circleDimensions(circle):
		#gives the radius and center of the circle:
		radius, center = cv2.minEnclosingCircle(image)
		#Calculations:
		circumference = 2 * math.PI * (radius*radius)
		#Check with the system:
		print("The radius of this circle is: " + radius + "\n")
		print("The Perimeter of this circle is: "+circumference+"\n")
	
	#ellipse = weird circle
	def ellipseDimensions(ellipse):
		peri = perimeterMeasure(ellipse)
		#diameters/radii:
		(majRad, center) = cv2.minEnclosingCircle(ellipse)
		print("The perimeter of this ellipse is: " + peri+"\n")
		print("The major radius of this ellipse is: "+majRad+"\n")
	
	#if the shape is some kind of polygon(triangle --> decagon+) call this.
	def polygonDimensions(img, polygon, numVerts):
		i = 0; k = 0
		perimeter = perimeterMeasure(polygon)
		#find the vertexes:
		corners[numVerts]
		corners[] = cv2.goodFeaturesToTrack(polygon, numVerts, 0.1, 1
			[, corners[][, [, 2[, true[, 0.01]]]]])
		numLens = numVerts + (numVerts / 2)
		num = (numVerts/2) + 1
		distance[numLens]
		#find the individual lengths and diagonals of the polygon:
		for i in range(0, num):
			color = (255, 0, 0)
			font = 3;
			size = 2;
			#draw the different lengths/diagonals:
			for j in range(i, numVerts):
				line = cv2.line(img, corners[i],corners[j],color, size)
				org = (corners[i]+corners[j])/2
				distance[k] = cv2.arcLength(line, false)
				cv2.putText(img, distance[k], org, font, size, color, 1, LINE_8, false)
				k+= 1

	#for all perimeters. Can call wherever and on whatever shape there is.
	def perimeterMeasure(shape):
		perimeter = cv2.arcLength(shape, true);
		return perimeter;
