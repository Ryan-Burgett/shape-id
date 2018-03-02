#	Imports
from PIL import Image
import cv2
import argparse
import pytesseract
import os

#	Construct an argument parser
ap = argparse.ArgumentParser()
#	Creates our args
ap.add_argument("-i", "--image", required=True, help = "path to the image to be parsed")
ap.add_argument("-p", "--preprocess", type=str, default="thresh", help = "type of preprocessing to be performed")
#	Parse our args
args = vars(ap.parse_args())
#	Process our image
image = cv2.imread(args["image"])
grayedImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#	Determine which preprocessing strategy to use
if args["preprocess"] == "thresh":
	grayedImage = cv2.threshold(grayedImage, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
elif args["preprocess"] == "blur":
	grayedImage = cv2.medianBlur(grayedImage, 3)
#	Save our processed image as a temp file so we can attempt OCR
tempfile = "{}.png".format(os.getpid())
cv2.imwrite(tempfile, grayedImage)
#	Attempt OCR and delete temp file
str = pytesseract.image_to_string(Image.open(tempfile))
os.remove(tempfile)
#	Display the image and print detected text
print(str)
cv2.imshow("Image", image)
cv2.imshow("Output", grayedImage)
cv2.waitKey(0)
cv2.destroyAllWindows()