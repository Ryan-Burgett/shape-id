#	Imports
import cv2
import argparse
import numpy as np
from basicdetector.tagdetector import TagDetector

#	Construct an argument parser
ap = argparse.ArgumentParser()
#	Creates arg -i for image and requires it for operation
ap.add_argument("-i", "--image", required=True, help = "path to the image to be parsed")
#	Parse our args
args = vars(ap.parse_args())

#	Pre-process our image
image = cv2.imread(args["image"])

#	Attempt to detect name tag in the image
tagD = TagDetector()
tag = tagD.detectTag(image)
