import cv2
import os
import numpy as np

subjects = ["Luke Safris", "Ryan Burgett", "Aidan", "Amy", "Chase", "Allison", "Steven", "Michael", "Alvin", "Toby", "Kelly", "Rodger", "Jordan",
"Ahmir", "Dilwyn", "Bernard", "Jerrard", "Wong", "George", "Sherman", "Kelvin", "Xiao", "Tre", "Braden", "Lucas", "Lyle", "Alexa", "Damion", "Herby",
"Ross", "Jules", "Dennis", "Dwight", "Darren", "Blake", "Aaron", "Mitchell", "Ghandi", "Hamilton", "Hawking", "Walter", "Miranda", "Debra", "Shalysa", "Simba", "Washington",
"Mufasa", "Joey", "Chandler", "Landon", "Kyle"]

def detect_face(img):
    #load LBP face Detector
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('cascades/lbpcascade_frontalface.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5)
    if(len(faces)== 0):
        return None,None

    x, y, w, h = faces[0]

    return gray[y:y+w, x:x+h], faces[0]

def prep_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []

    for dir_name in dirs:
        if not dir_name.startswith("s"):
            continue

        label = int(dir_name.replace("s",""))
        subject_dir_path = data_folder_path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)

        for image_name in subject_images_names:
            if image_name.startswith("."):
                continue

            image_path = subject_dir_path + "/" + image_name
            image = cv2.imread(image_path)

            cv2.imshow("Training on image...", image)
            cv2.waitKey(100)

            face, rect = detect_face(image)

            if face is not None:
                faces.append(face)
                labels.append(label)

            cv2.destroyAllWindows()
            cv2.waitKey(1)
            cv2.destroyAllWindows()

    return faces, labels

print("Preparing Data....")
faces, labels = prep_data("Images")
print("Data Prepared")

print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

face_recognizer = ecoginizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))


def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def predict(test_img):
    img = test_img.copy()
    face, rect = detect_face(img)
    label = face_recognizer.predict(face)
    print(label)
    label_text = subjects[label[0]]

    draw_rectangle(img, rect)
    draw_text(img, label_text, rect[0], rect[1]-5)

    return img

print("Predicting images...")

#load test Images
test_img1 = cv2.imread("test-data/test1.jpg")

if test_img1 is None:
    print("img 1 not found")

test_img2 = cv2.imread("test-data/test2.jpg")

if test_img1 is None:
    print("img 2 not found")

#Make a prediction
predicted_img1 = predict(test_img1)
predicted_img2 = predict(test_img2)

print("Prediction complete")

cv2.imshow(predicted_img1)
cv2.imshow(predicted_img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
