import cv2
import numpy as np
import pyttsx3
import face_recognition
import os
# from pyfirmata import Arduino
path = 'images'
#board = Arduino('COM7')
images = []
names = []
myList = os.listdir(path)
print(myList)
for mem in myList:
    curimg=cv2.imread(f'{path}/{mem}')
    images.append(curimg)
    names.append(os.path.splitext(mem)[0])
print(names)


engine = pyttsx3.init()

print(engine.getProperty('volume'))
voices = engine.getProperty('voices')
engine.setProperty('voice',voices[1].id)

def findEncodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

encodings = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(1)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

#this is for arduino , not necessary

 #   board.digital[9].write(0)
  #  board.digital[11].write(0)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodings,encodeFace)
        faceDis = face_recognition.face_distance(encodings,encodeFace)
        matchIndex = np.argmin(faceDis)

        print(faceDis)
        # when program will detect face it will greet him/her

        if matches[matchIndex]:
            name = names[matchIndex]
            y1,x2,y2,x1= faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),2)
            engine.say("Namaste"+name)
            engine.runAndWait()
        
    cv2.imshow('camera',img)
    cv2.waitKey(1)


