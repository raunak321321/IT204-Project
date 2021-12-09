import cv2
import numpy as np
import face_recognition
import os

path = 'ImageBasic'
images = []  # store all the images present inside the path
imageName = []  # store the names of all those images

myList = os.listdir(path)
# print(myList)
for im in myList:
    curImg = cv2.imread(f'{path}/{im}')
    images.append(curImg)
    imageName.append(os.path.splitext(im)[0])

# print(imageName)
# print(images)
def findEncodings(imagess):
    encodeList = []
    for img in imagess:
        currImg = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(currImg)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = findEncodings(images)
# print(len(encodeListKnown))

cap = cv2.VideoCapture(0)

while True:
    success,img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25) # image,pixelSize,,Scales
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    facesCurrFrame = face_recognition.face_locations(imgS)
    encodingsCurrFrame = face_recognition.face_encodings(imgS,facesCurrFrame)
    
    for encodingFace,faceLoc in zip(encodingsCurrFrame,facesCurrFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodingFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodingFace)

        matchIndex = np.argmin(faceDis)
        if(matches[matchIndex]):
            name = imageName[matchIndex].upper()    
            print(name)
    print()
