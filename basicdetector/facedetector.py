#	Imports
import cv2

class FaceDetector:
	def __init__(self):
		pass
	#	Attempt face detection
	def detectFace(self, image, cascade):
		return cascade.detectMultiScale(
			image,
			scaleFactor = 1.05,
			minNeighbors = 10,
			minSize = (30, 30),
			flags = cv2.CASCADE_SCALE_IMAGE
		)
	#	Attempt eye detection
	def detectEyes(self, image, cascade):
		return cascade.detectMultiScale(image, 1.1, 3)
	#	Prepare training data for face recognition, Ignore this for now
	def readyTrainingData(path, cascade):
		dirList = os.listdir(path)
		
		faces = []
		identifiers = []
		
		for dir in dirList:
			if not dir.startswith("face"):
				continue;
			
			id = int(dir.replace("face",""))
			subdir = path + "/" + dir
			images = os.listdir(subdir)
			
			for imageName in images:
				if image.startswith("."):
					continue;
				
				imageDir = subdir + "/" + imageName
				image = cv2.imread(imageDir)
				cv2.imshow("Training", image)
				cv2.waitKey(80)
				face, rect = detectFace(image, cascade)
				
				if face is not None:
					faces.append(face)
					identifiers.append(id)
		
		cv2.destroyAllWindows()
		cv2.waitKey(1)
		cv2.destroyAllWindows()
		
		return faces, identifiers
