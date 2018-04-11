

import cv2
import os
import numpy as np
import pdb

DEBUG = 0;

if DEBUG == 1:
    pdb.set_trace()

subjects = ["Luke Safris", "Ryan Burgett", "Aidan", "Amy", "Chase", "Allison", "Steven", "Michael", "Alvin", "Toby", "Kelly", "Rodger", "Jordan",
            "Ahmir", "Dilwyn", "Bernard", "Jerrard", "Wong", "George", "Sherman", "Kelvin", "Xiao", "Tre", "Braden", "Lucas", "Lyle", "Alexa", "Damion", "Herby",
            "Ross", "Jules", "Dennis", "Dwight", "Darren", "Blake", "Aaron", "Mitchell", "Ghandi", "Hamilton", "Hawking", "Walter", "Miranda", "Debra", "Shalysa", "Simba", "Washington",
            "Mufasa", "Joey", "Chandler", "Landon", "Kyle", "Guy Fieri", "Mr. Worldwide"]

def detect_face(img):
    #load LBP face Detector
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('cascades/lbpcascade_frontalface.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5)
    if(len(faces)== 0):
        return None,None

    x, y, w, h = faces[0]

    return gray[y:y+w, x:x+h], faces[0]

face_recognizer = ecoginizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("model.xml")




def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (226,9,9), 2)

def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (226,9,9), 2)

def predict(test_img):
    img = test_img.copy()
    if img is None:
        print("Image not found")

    face, rect = detect_face(img)

    if face is None:
        print("Face not detected")

    label = face_recognizer.predict(face)
    print(label)
    label_text = subjects[label[0]]

    draw_rectangle(img, rect)
    draw_text(img, label_text, rect[0], rect[1]-5)

    return img

print("Predicting images...")

#load test Images
test_img = cv2.imread("test-data/test.jpg")

if test_img is None:
    print("img 2 not found")

#Make a prediction
#predicted_img1 = predict(test_img1)
predicted_img = predict(test_img)

print("Prediction complete")

#cv2.imshow(predicted_img1)
cv2.imshow("Prediction", predicted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
