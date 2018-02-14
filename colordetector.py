#	Imports
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import cv2

class ColorDetector:
	def __init__(self):
		colors = OrderedDict({
			"red": (255,0,0),
			"green": (0, 255, 0),
			"blue": (0, 0, 255)
		})