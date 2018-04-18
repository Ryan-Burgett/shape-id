#	Imports
import cv2
import imutils
import argparse
import numpy as np
import os
import sys
from basicdetector.shapedetector import ShapeDetector
from basicdetector.circledetector import CircleDetector
from basicdetector.colordetector import ColorDetector

#	Construct an argument parser
ap = argparse.ArgumentParser()
#	Creates arg -i for image and requires it for operation
ap.add_argument("-i", "--image", required=True, help = "path to the image to be parsed")
ap.add_argument("--color", help="include the color of the contour", action="store_true")
ap.add_argument("--area", help="include the area of the contour", action="store_true")
ap.add_argument("--filter", help="Use advanced filtering method for noisy images", action="store_true")
ap.add_argument("-w", "--width", required=True, help = "width to use for the resized image. MUST BE THE LAST ARGUMENT")
#	Parse our args
args = vars(ap.parse_args())

#	Adjust the filepath to our new temp file 
#	Consoliadte the raw arguments to a string, ignoring the width
rawArgs = ""
for x in range(1, len(sys.argv)):
	if sys.argv[x] == "-w" or sys.argv[x] == "--width":
		break
	if sys.argv[x] == "-i" or sys.argv[x] == "--image":
		sys.argv[x+1] = "temp.png"
	rawArgs += sys.argv[x] + " "

#	Resize our image with the given width, assuming the width is not more than twice the default width
image = cv2.imread(args["image"])
h, w = image.shape[:2]
if int(args["width"]) < (w * 2):
	w = int(args["width"])
resized = imutils.resize(image, width = w)

#	Write the resized image to a temp file
cv2.imwrite('temp.png',resized)
#	Format and call our command for detect_shapes
command = "python detect_shapes.py {}".format(rawArgs)
os.system(command)
#	Delete the temp image
os.remove('temp.png')