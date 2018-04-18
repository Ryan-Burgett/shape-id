#	Imports
import cv2
import imutils
import matplotlib.patches as patches
import numpy as np
import os
import pytesseract
from PIL import Image
from skimage import measure
from skimage.measure import regionprops

class TagDetector:
	def __init__(self):
		pass
		
	def detectTag(self, image):
		
		ht, wd, cn = image.shape
		
		w = wd
		if int(wd > 800):
			w = 600
		resized = imutils.resize(image, width = w)
		
		blurFactor = int(wd*.005)
		if blurFactor % 2 == 0:
			blurFactor = blurFactor + 1
		
		bilateralImage = cv2.bilateralFilter(resized, 2*blurFactor+1, 5*blurFactor, blurFactor)
		grayedImage = cv2.cvtColor(bilateralImage, cv2.COLOR_BGR2GRAY)
		blurredImage = cv2.GaussianBlur(grayedImage,(blurFactor,blurFactor),0)
		#binaryImage = cv2.threshold(blurredImage, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
		binaryImage = cv2.adaptiveThreshold(blurredImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 2)
		
		cv2.imshow("Image", image)
		cv2.waitKey(0)
		cv2.imshow("Image", blurredImage)
		cv2.waitKey(0)
		cv2.imshow("Image", binaryImage)
		cv2.waitKey(0)
		
		labels = measure.label(binaryImage)

		for r in regionprops(labels):
			if r.area < (w * 0.035)**2:
				continue
			topRow, leftCol, botRow, rightCol = r.bbox
			boxWidth = rightCol - leftCol
			boxHeight = botRow - topRow
			ratio = boxHeight / boxWidth
			if ratio > 1.25 or ratio < 0.3:
				continue
			cv2.rectangle(image, (leftCol,topRow), (rightCol,botRow), (0,255,0), 2)
			
		cv2.imshow("Image", image)
		cv2.waitKey(0)
		
		for r in regionprops(labels):
			#	adjust this value for testing
			if r.area < (w * 0.04)**2:
				continue
			
			topRow, leftCol, botRow, rightCol = r.bbox
			boxWidth = rightCol - leftCol
			boxHeight = botRow - topRow
			ratio = boxHeight / boxWidth
			
			if ratio > 1.25 or ratio < 0.3 :
				continue
				
			roi = binaryImage[topRow:botRow, leftCol:rightCol]
			#roi = image[topRow:botRow, leftCol:rightCol]
			roi = cv2.bitwise_not(roi)
			roi = cv2.resize(roi, None, fx=4, fy=4, interpolation = cv2.INTER_CUBIC)
			
			boxHeight = boxHeight * 4
			boxWidth = boxWidth * 4
			
			cropped = roi[int(boxHeight/5):int(4*boxHeight/5),int(boxWidth/5):int(4*boxWidth/5)]
			
			morphKernel = np.ones((3,3),np.uint8)
			roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, morphKernel)

			cv2.imshow("Image", roi)
			cv2.waitKey(0)
			#	Save our processed image as a temp file so we can attempt OCR
			tempfile = "{}.png".format(os.getpid())
			cv2.imwrite(tempfile, roi)
			#	Attempt OCR and delete temp file
			str = pytesseract.image_to_string(Image.open(tempfile))
			if not str == "":
				print(str)
			else:
				tempfile = "{}.png".format(os.getpid())
				cv2.imwrite(tempfile, cropped)
				#	Attempt OCR and delete temp file
				str = pytesseract.image_to_string(Image.open(tempfile))
				cv2.imshow("Image", cropped)
				cv2.waitKey(0)
			if not str == "":
				print(str)
				return str
			else:
				print("No name recognized in this region. Moving on.")
			os.remove(tempfile)
			
		print("Complete")
		return "unknown"